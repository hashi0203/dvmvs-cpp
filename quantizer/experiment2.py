import cv2
import numpy as np
import torch
from path import Path
from tqdm import tqdm

from config import Config
from dataset_loader import PreprocessImage, load_image
from model import FeatureExtractor, FeatureShrinker, CostVolumeEncoder, LSTMFusion, CostVolumeDecoder
from keyframe_buffer2 import KeyframeBuffer
from utils import cost_volume_fusion, save_results, visualize_predictions, InferenceTimer, get_non_differentiable_rectangle_depth_estimation, \
    get_warp_grid_for_cost_volume_calculation, pose_distance

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


def depth_fusion(depth_forward, depth_backward):
    height, width = depth_forward.shape
    depth = np.zeros_like(depth_forward)
    for i in range(height):
        for j in range(width):
            if depth_forward[i][j] == 0:
                depth[i][j] = depth_backward[i][j]
            elif depth_backward[i][j] == 0:
                depth[i][j] = depth_forward[i][j]
            else:
                depth[i][j] = (depth_forward[i][j] + depth_backward[i][j]) / 2
    return depth

def predict_depths(depth, pose, next_pose, K, warp_grid):
    height, width = depth.shape
    warp_grid = warp_grid.cpu().numpy().squeeze()

    extrinsic2 = next_pose.dot(np.linalg.inv(pose))
    # extrinsic2 = np.linalg.inv(next_pose).dot(pose)
    R = extrinsic2[0:3, 0:3]
    t = extrinsic2[0:3, 3]

    Kt = K.dot(t)
    K_R_Kinv = K.dot(R).dot(np.linalg.inv(K))
    K_R_Kinv_UV = K_R_Kinv.dot(warp_grid)

    warping = K_R_Kinv_UV + (Kt.reshape(-1, 1) / depth.reshape(1, -1))
    warping = warping.transpose()
    warping = warping[:, 0:2] / (warping[:, 2:] + 1e-8)
    warping = warping.reshape(height, width, 2)

    ans = np.full((height, width), False)
    # inv_warping = np.zeros((height, width, 3))
    # inv_warping = np.zeros_like(warping)
    for i in range(height):
        for j in range(width):
            x, y = warping[i][j]
            ix, iy = int(x), int(y)
            for dx in range(2):
                for dy in range(2):
                    gx = ix + dx
                    gy = iy + dy
                    if 0 <= gx < width and 0 <= gy < height:
                        # w = (1-abs(x-gx)) * (1-abs(y-gy))
                        # inv_warping[gy][gx][0] += w * j
                        # inv_warping[gy][gx][1] += w * i
                        # inv_warping[gy][gx][2] += w
                        ans[gy][gx] = True
    return ans
    # for i in range(height):
    #     for j in range(width):
    #         if inv_warping[i][j][2] != 0:
    #             inv_warping[i][j] /= inv_warping[i][j][2]

    # R = pose[:3,:3]
    # t = pose[:3,3]
    # Kt = K.dot(t)
    # KR_inv = np.linalg.inv(K.dot(R))
    # R_next = next_pose[:3,:3]
    # t_next = next_pose[:3,3]
    # KR_next = K.dot(R_next)
    # d_next = K[2:].dot(t_next)

    # next_depth = np.zeros((height, width))
    # for i in range(height):
    #     for j in range(width):
    #         if inv_warping[i][j][2] != 0:
    #             x, y = inv_warping[i][j][:2]
    #             ix, iy = int(x), int(y)
    #             d = 0
    #             ws = 0
    #             for dx in range(2):
    #                 for dy in range(2):
    #                     gx = ix + dx
    #                     gy = iy + dy
    #                     if 0 <= gx < width and 0 <= gy < height:
    #                         w = (1-abs(x-gx)) * (1-abs(y-gy))
    #                         d += w * depth[gy][gx]
    #                         ws += w
    #             if ws != 0:
    #                 XYZ = KR_inv.dot((d / ws) * inv_warping[i][j] - Kt)
    #                 next_depth[i][j] = d_next + KR_next[2].dot(XYZ)

    # return next_depth

# def infer(reference_feature, full_K_torch, keyframe_buffer, device):
#     half_K_torch = full_K_torch.clone().cuda()
#     half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

#     lstm_K_bottom = full_K_torch.clone().cuda()
#     lstm_K_bottom[:, 0:2, :] = lstm_K_bottom[:, 0:2, :] / 32.0

#     measurement_poses_torch = []
#     # measurement_images_torch = []
#     measurement_feature_halfs = []
#     measurement_frames = keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
#     for (measurement_pose, measurement_feature_half) in measurement_frames:
#     # for (measurement_pose, measurement_image) in measurement_frames:
#     #     measurement_image = preprocessor.apply_rgb(image=measurement_image,
#     #                                                scale_rgb=scale_rgb,
#     #                                                mean_rgb=mean_rgb,
#     #                                                std_rgb=std_rgb)
#     #     measurement_image_torch = torch.from_numpy(np.transpose(measurement_image, (2, 0, 1))).float().to(device).unsqueeze(0)
#         measurement_pose_torch = torch.from_numpy(measurement_pose).float().to(device).unsqueeze(0)
#         measurement_poses_torch.append(measurement_pose_torch)
#         measurement_feature_halfs.append(measurement_feature_half)
#         # measurement_images_torch.append(measurement_image_torch)


#     # measurement_feature_halfs = []
#     # for measurement_image_torch in measurement_images_torch:
#     #     measurement_feature_half, _, _, _ = feature_shrinker(*feature_extractor(measurement_image_torch))
#     #     measurement_feature_halfs.append(measurement_feature_half)

#     reference_feature_half, reference_feature_quarter, \
#         reference_feature_one_eight, reference_feature_one_sixteen = reference_feature

#     cost_volume = cost_volume_fusion(image1=reference_feature_half,
#                                      image2s=measurement_feature_halfs,
#                                      pose1=reference_pose_torch,
#                                      pose2s=measurement_poses_torch,
#                                      K=half_K_torch,
#                                      warp_grid=warp_grid,
#                                      min_depth=min_depth,
#                                      max_depth=max_depth,
#                                      n_depth_levels=n_depth_levels,
#                                      device=device,
#                                      dot_product=True)

#     skip0, skip1, skip2, skip3, bottom = cost_volume_encoder(features_half=reference_feature_half,
#                                                                 features_quarter=reference_feature_quarter,
#                                                                 features_one_eight=reference_feature_one_eight,
#                                                                 features_one_sixteen=reference_feature_one_sixteen,
#                                                                 cost_volume=cost_volume)

#     if previous_depth is not None:
#         depth_estimation = get_non_differentiable_rectangle_depth_estimation(reference_pose_torch=reference_pose_torch,
#                                                                              measurement_pose_torch=previous_pose,
#                                                                              previous_depth_torch=previous_depth,
#                                                                              full_K_torch=full_K_torch,
#                                                                              half_K_torch=half_K_torch,
#                                                                              original_height=Config.test_image_height,
#                                                                              original_width=Config.test_image_width)
#         depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
#                                                             scale_factor=(1.0 / 16.0),
#                                                             mode="nearest")
#     else:
#         depth_estimation = torch.zeros(size=(1, 1, int(Config.test_image_height / 32.0), int(Config.test_image_width / 32.0))).to(device)

#     lstm_state = lstm_fusion(current_encoding=bottom,
#                                 current_state=lstm_state,
#                                 previous_pose=previous_pose,
#                                 current_pose=reference_pose_torch,
#                                 estimated_current_depth=depth_estimation,
#                                 camera_matrix=lstm_K_bottom)

#     prediction, _, _, _, _ = cost_volume_decoder(reference_image_torch, skip0, skip1, skip2, skip3, lstm_state[0])


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
    warp_grid_depth = get_warp_grid_for_cost_volume_calculation(width=Config.test_image_width,
                                                                height=Config.test_image_height,
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

    frames = []
    tmp_frames = []
    frame_buffer = []
    predictions = []

    threshold = Config.test_image_width * Config.test_image_height * 0.8

    if evaluate:
        reference_depths = []
        depth_filenames = sorted((scene_folder / 'depth').files("*.png"))
    else:
        # if None the system will not be evaluated and errors will not be calculated
        reference_depths = None
        depth_filenames = None

    # frames = [0]
    # prev_pose = poses[0]
    # for i in range(1, min(len(poses), Config.n_test_frames)):
    #     if pose_distance(prev_pose, poses[i])[0] >= Config.test_keyframe_pose_distance:
    #         frames.append(i)
    #         prev_pose = poses[i]

    with torch.no_grad():
        # for i in tqdm(range(0, len(poses))):
        for i in range(len(poses)):
            reference_pose = poses[i]
            reference_image = load_image(image_filenames[i])

            # POLL THE KEYFRAME BUFFER
            response = keyframe_buffer.try_new_keyframe(reference_pose)
            if response == 2 or response == 4 or response == 5:
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

            # if reference_depths is not None:
            #     reference_depth = cv2.imread(depth_filenames[f], -1).astype(float) / 1000.0
            #     reference_depth = preprocessor.apply_depth(reference_depth)
            #     reference_depths.append(reference_depth)

            reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().to(device).unsqueeze(0)
            reference_pose_torch = torch.from_numpy(reference_pose).float().to(device).unsqueeze(0)

            if response == 0:
                reference_feature_half, _, _, _ = feature_shrinker(*feature_extractor(reference_image_torch))
                keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half)
                continue

            tmp_frames.append(i)
            frame_buffer.append((reference_pose_torch.clone(), reference_image_torch.clone()))
            if len(frame_buffer) < 4 and len(predictions) > 0:
                continue

            full_K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().to(device).unsqueeze(0)

            half_K_torch = full_K_torch.clone().cuda()
            half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

            lstm_K_bottom = full_K_torch.clone().cuda()
            lstm_K_bottom[:, 0:2, :] = lstm_K_bottom[:, 0:2, :] / 32.0

            inference_timer.record_start_time()

            # for f in [3, 1, 0, 2]:
            if len(predictions) == 0:
                frame_buffer *= 4
                tmp_frames *= 4

            f = 3
            tmp_previous_depth = None if previous_depth is None else previous_depth.clone()
            previous_depths = [tmp_previous_depth, tmp_previous_depth, None, tmp_previous_depth]
            tmp_previous_pose = None if previous_pose is None else previous_pose.clone()
            previous_poses = [tmp_previous_pose, tmp_previous_pose, None, tmp_previous_pose]
            tmp_lstm_states = None if lstm_state is None else (lstm_state[0].clone(), lstm_state[1].clone())
            lstm_states = [tmp_lstm_states, tmp_lstm_states, None, tmp_lstm_states]
            preds = [None] * 4
            nf = None
            while True:
                reference_pose_torch, reference_image_torch = frame_buffer[f]
                reference_feature_half, reference_feature_quarter, \
                reference_feature_one_eight, reference_feature_one_sixteen = feature_shrinker(*feature_extractor(reference_image_torch))
                keyframe_buffer.add_new_keyframe(reference_pose_torch.cpu().numpy().squeeze(), reference_feature_half)

                measurement_poses_torch = []
                # measurement_images_torch = []
                measurement_feature_halfs = []
                measurement_frames = keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
                for (measurement_pose, measurement_feature_half) in measurement_frames:
                # for (measurement_pose, measurement_image) in measurement_frames:
                #     measurement_image = preprocessor.apply_rgb(image=measurement_image,
                #                                                scale_rgb=scale_rgb,
                #                                                mean_rgb=mean_rgb,
                #                                                std_rgb=std_rgb)
                #     measurement_image_torch = torch.from_numpy(np.transpose(measurement_image, (2, 0, 1))).float().to(device).unsqueeze(0)
                    measurement_pose_torch = torch.from_numpy(measurement_pose).float().to(device).unsqueeze(0)
                    measurement_poses_torch.append(measurement_pose_torch)
                    measurement_feature_halfs.append(measurement_feature_half)
                    # measurement_images_torch.append(measurement_image_torch)


                # measurement_feature_halfs = []
                # for measurement_image_torch in measurement_images_torch:
                #     measurement_feature_half, _, _, _ = feature_shrinker(*feature_extractor(measurement_image_torch))
                #     measurement_feature_halfs.append(measurement_feature_half)


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

                if previous_depths[f] is not None:
                    depth_estimation = get_non_differentiable_rectangle_depth_estimation(reference_pose_torch=reference_pose_torch,
                                                                                        measurement_pose_torch=previous_poses[f],
                                                                                        previous_depth_torch=previous_depths[f],
                                                                                        full_K_torch=full_K_torch,
                                                                                        half_K_torch=half_K_torch,
                                                                                        original_height=Config.test_image_height,
                                                                                        original_width=Config.test_image_width)
                    depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
                                                                    scale_factor=(1.0 / 16.0),
                                                                    mode="nearest")
                else:
                    depth_estimation = torch.zeros(size=(1, 1, int(Config.test_image_height / 32.0), int(Config.test_image_width / 32.0))).to(device)

                cur_lstm_state = lstm_fusion(current_encoding=bottom,
                                        current_state=lstm_states[f],
                                        previous_pose=previous_poses[f],
                                        current_pose=reference_pose_torch,
                                        estimated_current_depth=depth_estimation,
                                        camera_matrix=lstm_K_bottom)

                prediction, _, _, _, _ = cost_volume_decoder(reference_image_torch, skip0, skip1, skip2, skip3, cur_lstm_state[0])
                if f == 3:
                    previous_depth = prediction.view(1, 1, Config.test_image_height, Config.test_image_width).clone()
                    previous_pose = reference_pose_torch.clone()
                    lstm_state = (cur_lstm_state[0].clone(), cur_lstm_state[1].clone())
                elif f == 1:
                    previous_depths[2] = prediction.view(1, 1, Config.test_image_height, Config.test_image_width).clone()
                    previous_poses[2] = reference_pose_torch.clone()
                    lstm_state = lstm_states[2] = (cur_lstm_state[0].clone(), cur_lstm_state[1].clone())
                elif f == 2:
                    lstm_state = (cur_lstm_state[0].clone(), cur_lstm_state[1].clone())

                inference_timer.record_end_time_and_elapsed_time()

                prediction = prediction.cpu().numpy().squeeze()
                preds[f] = prediction
                if f == 2 or len(predictions) == 0:
                    break

                if f == 3:
                    depth_forward = predict_depths(predictions[-1], poses[frames[-1]], poses[tmp_frames[1]], K, warp_grid_depth)
                    depth_backward = predict_depths(preds[3], poses[tmp_frames[3]], poses[tmp_frames[1]], K, warp_grid_depth)
                    # depth = depth_fusion(depth_forward, depth_backward)
                    n_fill = np.sum(np.logical_or(depth_forward, depth_backward))
                    print(n_fill)
                    if n_fill < threshold:
                        f = 1
                    else:
                        break
                elif f == 1:
                    depth_forward = predict_depths(predictions[-1], poses[frames[-1]], poses[tmp_frames[0]], K, warp_grid_depth)
                    depth_backward = predict_depths(preds[1], poses[tmp_frames[1]], poses[tmp_frames[0]], K, warp_grid_depth)
                    n_fill = np.sum(np.logical_or(depth_forward, depth_backward))
                    print(n_fill)
                    if n_fill < threshold:
                        f = 0

                    depth_forward = predict_depths(preds[1], poses[tmp_frames[1]], poses[tmp_frames[2]], K, warp_grid_depth)
                    depth_backward = predict_depths(preds[3], poses[tmp_frames[3]], poses[tmp_frames[2]], K, warp_grid_depth)
                    n_fill = np.sum(np.logical_or(depth_forward, depth_backward))
                    print(n_fill)
                    if n_fill < threshold and f == 1:
                        f = 2
                    elif n_fill < threshold:
                        nf = 2
                    elif f == 1:
                        break
                else:
                    if nf is None:
                        break
                    f = nf

            for f in range(4):
                if preds[f] is not None:
                    frames.append(tmp_frames[f])
                    predictions.append(preds[f])

                    if reference_depths is not None:
                        reference_depth = cv2.imread(depth_filenames[frames[-1]], -1).astype(float) / 1000.0
                        reference_depth = preprocessor.apply_depth(reference_depth)
                        print('%d: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n' % (frames[-1], *compute_errors(reference_depth, predictions[-1], max_depth)))

                    with open('results-exp2/frame-0%s.pose.txt' % (image_filenames[frames[-1]].split("/")[-1][:-4]), 'w') as fout:
                        for p in frame_buffer[f][0].cpu().numpy().squeeze():
                            fout.write("%.7e %.7e %.7e %.7e\n" % tuple(p))
                    cv2.imwrite('results-exp2/frame-0%s.color.png' % (image_filenames[frames[-1]].split("/")[-1][:-4]), cv2.resize(cv2.imread(image_filenames[frames[-1]], cv2.IMREAD_COLOR), dsize=predictions[-1].shape[::-1]))
                    cv2.imwrite('results-exp2/frame-0%s.depth.png' % (image_filenames[frames[-1]].split("/")[-1][:-4]), (predictions[-1] * 1000).astype(np.uint16))

            tmp_frames = []
            frame_buffer = []


            # if depth_forward is not None:
            #     depth_backward = predict_depths(prediction, poses[i], poses[frames[i-1]], K, warp_grid_depth)
            #     depth = depth_fusion(depth_forward, depth_backward)
            #     cv2.imwrite('results-exp/%s' % (image_filenames[frames[i-1]].split("/")[-1]), (depth * 25).astype(np.uint8))
            #     predictions.append(depth)

                # if reference_depths is not None:
                #     reference_depth = cv2.imread(depth_filenames[frames[i-1]], -1).astype(float) / 1000.0
                #     reference_depth = preprocessor.apply_depth(reference_depth)
                #     print('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n' % compute_errors(reference_depth, depth, max_depth))


            # if reference_depths is not None:
            #     reference_depth = cv2.imread(depth_filenames[f], -1).astype(float) / 1000.0
            #     reference_depth = preprocessor.apply_depth(reference_depth)
            #     print('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n' % compute_errors(reference_depth, prediction, max_depth))

            # predictions.append(prediction)

            # if Config.test_visualize:
            #     visualize_predictions(numpy_reference_image=reference_image,
            #                           numpy_measurement_image=measurement_image,
            #                           numpy_predicted_depth=prediction,
            #                           normalization_mean=mean_rgb,
            #                           normalization_std=std_rgb,
            #                           normalization_scale=scale_rgb,
            #                           depth_multiplier_for_visualization=5000)

        inference_timer.print_statistics()



        save_results(predictions=predictions,
                     groundtruths=reference_depths,
                     system_name=system_name,
                     scene_name=scene,
                     save_folder=".")

    return frames, predictions


if __name__ == '__main__':
    predict(evaluate=True)
