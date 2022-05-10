import cv2
import numpy as np
import torch
from path import Path
from tqdm import tqdm

from config import Config
from dataset_loader import PreprocessImage, load_image
from model import FeatureExtractor, FeatureShrinker, CostVolumeEncoder, LSTMFusion, CostVolumeDecoder
from keyframe_buffer import KeyframeBuffer
from utils import cost_volume_fusion, save_results, visualize_predictions, InferenceTimer, get_non_differentiable_rectangle_depth_estimation, \
    get_warp_grid_for_cost_volume_calculation

def compute_errors(gt, pred, max_depth=np.inf):
    valid1 = gt >= 0.5
    valid2 = gt <= max_depth
    valid = valid1 & valid2
    gt = gt[valid]
    pred = pred[valid]

    n_valid = np.float32(len(gt))
    if n_valid == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    differences = gt - pred
    abs_differences = np.abs(differences)
    squared_differences = np.square(differences)
    abs_error = np.mean(abs_differences)
    abs_relative_error = np.mean(abs_differences / gt)
    abs_inverse_error = np.mean(np.abs(1 / gt - 1 / pred))
    squared_relative_error = np.mean(squared_differences / gt)
    rmse = np.sqrt(np.mean(squared_differences))
    ratios = np.maximum(gt / pred, pred / gt)
    n_valid = np.float32(len(ratios))
    ratio_125 = np.count_nonzero(ratios < 1.25) / n_valid
    ratio_125_2 = np.count_nonzero(ratios < 1.25 ** 2) / n_valid
    ratio_125_3 = np.count_nonzero(ratios < 1.25 ** 3) / n_valid
    return abs_error, abs_relative_error, abs_inverse_error, squared_relative_error, rmse, ratio_125, ratio_125_2, ratio_125_3


def grid_sample(image, grid):
    height, width = image.shape
    warped_image = np.zeros_like(image)
    for j in range(height):
        for k in range(width):
            x = (grid[j][k][0] + 1) * (width - 1) / 2.0
            y = (grid[j][k][1] + 1) * (height - 1) / 2.0
            y_int = int(y)
            x_int = int(x)
            ys = [y_int, y_int + 1]
            xs = [x_int, x_int + 1]
            dys = [y - ys[0], ys[1] - y]
            dxs = [x - xs[0], xs[1] - x]
            for yi in range(2):
                for xi in range(2):
                    val = 0 if (ys[yi] < 0 or height-1 < ys[yi] or xs[xi] < 0 or width-1 < xs[xi]) else image[ys[yi]][xs[xi]]
                    warped_image[j][k] += dys[1-yi] * dxs[1-xi] * val
    return warped_image


def predict_features(depth, image, pose, next_pose, K, warp_grid):
    height, width = image.shape
    width_normalizer = width / 2.0
    height_normalizer = height / 2.0
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    warp_grid = warp_grid.cpu().numpy().squeeze()

    # extrinsic2 = pose.dot(np.linalg.inv(next_pose))
    # # extrinsic2 = np.linalg.inv(pose).dot(next_pose)
    # R = extrinsic2[0:3, 0:3]
    # t = extrinsic2[0:3, 3]

    # Kt = K.dot(t)
    # K_R_Kinv = K.dot(R).dot(np.linalg.inv(K))
    # K_R_Kinv_UV = K_R_Kinv.dot(warp_grid)

    # warping = K_R_Kinv_UV + (Kt.reshape(-1, 1) / depth.reshape(1, -1))
    # warping = warping.transpose()
    # warping = warping[:, 0:2] / (warping[:, 2:] + 1e-8)
    # warping = warping.reshape(height, width, 2)
    # warping[:, :, 0] = (warping[:, :, 0] - width_normalizer) / width_normalizer
    # warping[:, :, 1] = (warping[:, :, 1] - height_normalizer) / height_normalizer

    extrinsic2 = pose.dot(np.linalg.inv(next_pose))
    # extrinsic2 = np.linalg.inv(next_pose).dot(pose)
    R = extrinsic2[0:3, 0:3]
    t = extrinsic2[0:3, 3]
    Kt = K.dot(t)
    K_R_Kinv = K.dot(R).dot(np.linalg.inv(K))
    K_R_Kinv_UV = K_R_Kinv.dot(warp_grid)
    depth_rate = (1.0 - (Kt[2] / depth.reshape(-1))) / K_R_Kinv_UV[2]
    warping = depth_rate * K_R_Kinv_UV[:2] + (Kt[:2].reshape(-1, 1) / depth.reshape(1, -1))
    warping = warping.transpose()
    warping = warping.reshape(height, width, 2)
    warping[:, :, 0] = (warping[:, :, 0] - width_normalizer) / width_normalizer
    warping[:, :, 1] = (warping[:, :, 1] - height_normalizer) / height_normalizer

    return grid_sample(image, warping)

def predict(evaluate):
    dataset_name = Config.test_online_scene_path.split("/")[-2]
    system_name = "keyframe_{}_{}_{}_{}_dvmvs_fusionnet_online".format(dataset_name,
                                                                       Config.test_image_width,
                                                                       Config.test_image_height,
                                                                       Config.test_n_measurement_frames)

    print("Predicting with System:", system_name)
    print("# of Measurement Frames:", Config.test_n_measurement_frames)

    device = torch.device("cuda")
    feature_extractor = FeatureExtractor()
    feature_shrinker = FeatureShrinker()
    cost_volume_encoder = CostVolumeEncoder()
    lstm_fusion = LSTMFusion()
    cost_volume_decoder = CostVolumeDecoder()

    feature_extractor = feature_extractor.to(device)
    feature_shrinker = feature_shrinker.to(device)
    cost_volume_encoder = cost_volume_encoder.to(device)
    lstm_fusion = lstm_fusion.to(device)
    cost_volume_decoder = cost_volume_decoder.to(device)

    model = [feature_extractor, feature_shrinker, cost_volume_encoder, lstm_fusion, cost_volume_decoder]

    for i in range(len(model)):
        try:
            checkpoint = sorted(Path(Config.fusionnet_test_weights).files())[i]
            weights = torch.load(checkpoint)
            model[i].load_state_dict(weights)
            model[i].eval()
            print("Loaded weights for", checkpoint)
        except Exception as e:
            print(e)
            print("Could not find the checkpoint for module", i)
            exit(1)

    feature_extractor = model[0]
    feature_shrinker = model[1]
    cost_volume_encoder = model[2]
    lstm_fusion = model[3]
    cost_volume_decoder = model[4]

    warp_grid = get_warp_grid_for_cost_volume_calculation(width=int(Config.test_image_width / 2),
                                                          height=int(Config.test_image_height / 2),
                                                          device=device)

    scale_rgb = 255.0
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]

    min_depth = 0.25
    max_depth = 20.0
    n_depth_levels = 64

    scene_folder = Path(Config.test_online_scene_path)

    scene = scene_folder.split("/")[-1]
    print("Predicting for scene:", scene)

    keyframe_buffer = KeyframeBuffer(buffer_size=Config.test_keyframe_buffer_size,
                                     keyframe_pose_distance=Config.test_keyframe_pose_distance,
                                     optimal_t_score=Config.test_optimal_t_measure,
                                     optimal_R_score=Config.test_optimal_R_measure,
                                     store_return_indices=False)

    K = np.loadtxt(scene_folder / 'K.txt').astype(np.float32)
    poses = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
    image_filenames = sorted((scene_folder / 'images').files("*.png"))

    inference_timer = InferenceTimer()

    lstm_state = None
    previous_depth = None
    previous_pose = None

    predictions = [[],[]]
    datas = []
    reference_features_half = []
    predicted_reference_features_half = []
    iis = []

    if evaluate:
        reference_depths = []
        depth_filenames = sorted((scene_folder / 'depth').files("*.png"))
        error_names = ['abs_error', 'abs_relative_error', 'abs_inverse_error',
                       'squared_relative_error', 'rmse', 'ratio_125', 'ratio_125_2', 'ratio_125_3']
        with open('errors.txt', 'w') as f:
            f.writelines(', '.join(error_names) + '\n')
        with open('errors-0.txt', 'w') as f:
            f.writelines(', '.join(error_names) + '\n')
        with open('errors-1.txt', 'w') as f:
            f.writelines(', '.join(error_names) + '\n')
    else:
        # if None the system will not be evaluated and errors will not be calculated
        reference_depths = None
        depth_filenames = None

    # for ii in range(10):
    #     reference_image = load_image(image_filenames[ii])
    #     grid = get_warp_grid_for_cost_volume_calculation(width=Config.org_image_width,
    #                                             height=Config.org_image_height,
    #                                             device=device)
    #     pred = np.zeros_like(reference_image)
    #     for jj in range(3):
    #         pred[:,:,jj] = predict_features(reference_image[:,:,jj], poses[ii], poses[ii+1], K, grid)
    #     cv2.imwrite('test/%s-pred.png' % (image_filenames[ii+1].split("/")[-1][:-4]), pred.astype(np.uint8))
    #     cv2.imwrite('test/%s' % (image_filenames[ii].split("/")[-1]), reference_image.astype(np.uint8))

    with torch.no_grad():
        # for i in tqdm(range(0, min(len(poses), Config.n_test_frames))):
        for i in range(0, min(len(poses), Config.n_test_frames)):
            reference_pose = poses[i]
            reference_image = load_image(image_filenames[i])

            # POLL THE KEYFRAME BUFFER
            response = keyframe_buffer.try_new_keyframe(reference_pose, reference_image)
            print("%s: %d" % (image_filenames[i].split("/")[-1], response))
            if response == 0 or response == 2 or response == 4 or response == 5:
                continue
            elif response == 3:
                previous_depth = None
                previous_pose = None
                lstm_state = None
                continue

            preprocessor = PreprocessImage(K=K,
                                        old_width=reference_image.shape[1],
                                        old_height=reference_image.shape[0],
                                        new_width=Config.test_image_width,
                                        new_height=Config.test_image_height,
                                        distortion_crop=Config.test_distortion_crop,
                                        perform_crop=Config.test_perform_crop)

            reference_image = preprocessor.apply_rgb(image=reference_image,
                                                    scale_rgb=scale_rgb,
                                                    mean_rgb=mean_rgb,
                                                    std_rgb=std_rgb)

            reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().to(device).unsqueeze(0)
            reference_pose_torch = torch.from_numpy(reference_pose).float().to(device).unsqueeze(0)

            full_K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().to(device).unsqueeze(0)

            half_K_torch = full_K_torch.clone().cuda()
            half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

            lstm_K_bottom = full_K_torch.clone().cuda()
            lstm_K_bottom[:, 0:2, :] = lstm_K_bottom[:, 0:2, :] / 32.0

            measurement_poses_torch = []
            measurement_images_torch = []
            measurement_frames = keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
            for (measurement_pose, measurement_image) in measurement_frames:
                measurement_image = preprocessor.apply_rgb(image=measurement_image,
                                                        scale_rgb=scale_rgb,
                                                        mean_rgb=mean_rgb,
                                                        std_rgb=std_rgb)
                measurement_image_torch = torch.from_numpy(np.transpose(measurement_image, (2, 0, 1))).float().to(device).unsqueeze(0)
                measurement_pose_torch = torch.from_numpy(measurement_pose).float().to(device).unsqueeze(0)
                measurement_images_torch.append(measurement_image_torch)
                measurement_poses_torch.append(measurement_pose_torch)

            inference_timer.record_start_time()

            measurement_feature_halfs = []
            for measurement_image_torch in measurement_images_torch:
                measurement_feature_half, _, _, _ = feature_shrinker(*feature_extractor(measurement_image_torch))
                measurement_feature_halfs.append(measurement_feature_half)


            reference_feature_half, reference_feature_quarter, \
            reference_feature_one_eight, reference_feature_one_sixteen = feature_shrinker(*feature_extractor(reference_image_torch))

            for j in range(2):
                if j == 0:
                    datas.append(reference_image_torch.cpu().numpy().squeeze())
                    reference_features_half.append(reference_feature_half.cpu().numpy().squeeze())
                elif len(reference_features_half) > 1:
                    for jj in range(len(reference_features_half[-2])):
                        # reference_feature_half[0][jj] = torch.from_numpy(predict_features(predictions[0][-2], reference_features_half[-2][jj], previous_pose.cpu().numpy().squeeze(), reference_pose, K, warp_grid)).float().to(device)
                        if len(predicted_reference_features_half) > 0:
                            reference_feature_half[0][jj] = torch.from_numpy(predict_features(predictions[0][-2], predicted_reference_features_half[-1][jj], previous_pose.cpu().numpy().squeeze(), reference_pose, K, warp_grid)).float().to(device)
                        else:
                            reference_feature_half[0][jj] = torch.from_numpy(predict_features(predictions[0][-2], reference_features_half[-2][jj], previous_pose.cpu().numpy().squeeze(), reference_pose, K, warp_grid)).float().to(device)
                        predicted_reference_features_half.append(reference_feature_half.cpu().numpy().squeeze())
                else:
                    continue


                cost_volume = cost_volume_fusion(image1=reference_feature_half,
                                                image2s=measurement_feature_halfs,
                                                pose1=reference_pose_torch,
                                                pose2s=measurement_poses_torch,
                                                K=half_K_torch,
                                                warp_grid=warp_grid,
                                                min_depth=min_depth,
                                                max_depth=max_depth,
                                                n_depth_levels=n_depth_levels,
                                                device=device,
                                                dot_product=True)

                skip0, skip1, skip2, skip3, bottom = cost_volume_encoder(features_half=reference_feature_half,
                                                                        features_quarter=reference_feature_quarter,
                                                                        features_one_eight=reference_feature_one_eight,
                                                                        features_one_sixteen=reference_feature_one_sixteen,
                                                                        cost_volume=cost_volume)

                if previous_depth is not None:
                    depth_estimation = get_non_differentiable_rectangle_depth_estimation(reference_pose_torch=reference_pose_torch,
                                                                                        measurement_pose_torch=previous_pose,
                                                                                        previous_depth_torch=previous_depth,
                                                                                        full_K_torch=full_K_torch,
                                                                                        half_K_torch=half_K_torch,
                                                                                        original_height=Config.test_image_height,
                                                                                        original_width=Config.test_image_width)
                    depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
                                                                    scale_factor=(1.0 / 16.0),
                                                                    mode="nearest")
                else:
                    depth_estimation = torch.zeros(size=(1, 1, int(Config.test_image_height / 32.0), int(Config.test_image_width / 32.0))).to(device)

                lstm_state = lstm_fusion(current_encoding=bottom,
                                        current_state=lstm_state,
                                        previous_pose=previous_pose,
                                        current_pose=reference_pose_torch,
                                        estimated_current_depth=depth_estimation,
                                        camera_matrix=lstm_K_bottom)

                prediction, _, _, _, _ = cost_volume_decoder(reference_image_torch, skip0, skip1, skip2, skip3, lstm_state[0])
                prediction = prediction.view(1, 1, Config.test_image_height, Config.test_image_width)
                predictions[j].append(prediction.cpu().numpy().squeeze())
                cv2.imwrite('results-%d/%s' % (j, image_filenames[i].split("/")[-1]), (prediction.cpu().numpy().squeeze() * 25).astype(np.uint8))

                if reference_depths is not None:
                    reference_depth = cv2.imread(depth_filenames[i], -1).astype(float) / 1000.0
                    reference_depth = preprocessor.apply_depth(reference_depth)
                    # reference_depths.append(reference_depth)
                    with open('errors-%d.txt' % j, 'a') as f:
                        f.writelines('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n' % compute_errors(reference_depth, prediction.cpu().numpy().squeeze(), max_depth))

            previous_depth = prediction
            previous_pose = reference_pose_torch
            # if i == 6:
            #     print(prediction[:,10])
            # break

            inference_timer.record_end_time_and_elapsed_time()

            # prediction = prediction.cpu().numpy().squeeze()
            # predictions.append(prediction)

            # with open('results/%s.txt' % image_filenames[i].split("/")[-1][:-4], 'w') as f:
            #     for ii in range(len(prediction)):
            #         f.writelines(' '.join(map(str, prediction[ii])))

            # cv2.imwrite('results/%s' % image_filenames[i].split("/")[-1], (prediction * 150).astype(np.uint8))

            # if reference_depths is not None:
            #     reference_depth = cv2.imread(depth_filenames[i], -1).astype(float) / 1000.0
            #     reference_depth = preprocessor.apply_depth(reference_depth)
            #     # reference_depths.append(reference_depth)
            #     with open('errors.txt', 'a') as f:
            #         f.writelines('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n' % compute_errors(reference_depth, prediction, max_depth))
            iis.append(i)

            if Config.test_visualize:
                visualize_predictions(numpy_reference_image=reference_image,
                                    numpy_measurement_image=measurement_image,
                                    numpy_predicted_depth=prediction,
                                    normalization_mean=mean_rgb,
                                    normalization_std=std_rgb,
                                    normalization_scale=scale_rgb,
                                    depth_multiplier_for_visualization=5000)

        inference_timer.print_statistics()
        print(iis)

        # with open('datas.txt', 'w') as f:
        #     f.writelines('abs_error\n')
        # for i in range(len(datas)-1):
        #     differences = datas[i+1] - datas[i]
        #     abs_differences = np.abs(differences)
        #     abs_error = np.mean(abs_differences)
        #     with open('datas.txt', 'a') as f:
        #         f.writelines('%.3f\n' % abs_error)

        with open('datas.txt', 'w') as f:
            f.writelines('psnr\n')
        for i in range(len(datas)-1):
            psnr = cv2.PSNR(datas[i], datas[i+1])
            with open('datas.txt', 'a') as f:
                f.writelines('%.3f\n' % psnr)

        for ii in range(20):
            for jj in range(5):
                cv2.imwrite('half_%d/%s' % (jj, image_filenames[iis[ii]].split("/")[-1]), (reference_features_half[ii][jj] + 128).astype(np.uint8))

        # for ii in range(20):
        #     for jj in range(5):
        #         pred = predict_features(predictions[ii], reference_features_half[ii][jj], poses[iis[ii]], poses[iis[ii+1]], K, warp_grid)
        #         print('%d: %.3f' % (iis[ii+1], cv2.PSNR(pred, reference_features_half[ii+1][jj])))
        #         cv2.imwrite('half_%d/%s-pred.png' % (jj, image_filenames[iis[ii+1]].split("/")[-1][:-4]), (pred + 128).astype(np.uint8))


        # save_results(predictions=predictions,
        #              groundtruths=reference_depths,
        #              system_name=system_name,
        #              scene_name=scene,
        #              save_folder=".")


if __name__ == '__main__':
    predict(evaluate=True)
