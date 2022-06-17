import cv2
import numpy as np
import torch
from path import Path
import os
import random
import matplotlib.pyplot as plt

from config import Config
from dataset_loader import PreprocessImage, load_image
from model import FeatureExtractor, FeatureShrinker, CostVolumeEncoder, LSTMFusion, CostVolumeDecoder
from keyframe_buffer import KeyframeBuffer
from utils import cost_volume_fusion, get_non_differentiable_rectangle_depth_estimation, get_warp_grid_for_cost_volume_calculation

def compute_errors(gt, pred):
    # MSE, RMSE
    valid1 = gt >= 0.5
    valid2 = gt <= 20.0
    valid = valid1 & valid2
    gt = gt[valid]
    pred = pred[valid]

    if len(gt) == 0:
        return np.nan, np.nan, np.nan

    differences = gt - pred
    squared_differences = np.square(differences)
    mse = np.mean(squared_differences)
    rmse = np.sqrt(mse)

    return mse, rmse

def resize_torch(arr, target_shape, device):
    new_shape = (target_shape[-1], target_shape[-2])
    return torch.from_numpy(np.array([cv2.resize(org, new_shape) for org in arr.cpu().numpy().reshape(-1, *arr.shape[-2:])])).float().to(device).view(target_shape)


def predict(param, device, model, K, poses, image_filenames, depth_filenames, warp_grid):
    feature_extractor = model[0]
    feature_shrinker = model[1]
    cost_volume_encoder = model[2]
    lstm_fusion = model[3]
    cost_volume_decoder = model[4]

    scale_rgb = 255.0
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]

    min_depth = 0.25
    max_depth = 20.0
    n_depth_levels = 64

    keyframe_buffer = KeyframeBuffer(buffer_size=Config.test_keyframe_buffer_size,
                                     keyframe_pose_distance=Config.test_keyframe_pose_distance,
                                     optimal_t_score=Config.test_optimal_t_measure,
                                     optimal_R_score=Config.test_optimal_R_measure,
                                     store_return_indices=False)

    lstm_state = [None, None]
    previous_depth = [None, None]
    previous_pose = [None, None]
    previous_idx = [-1, -1]

    methods = []
    mses = []
    rmses = []

    # save_path = "%s-%d/%s" % (method_name, param, test_dataset_name)
    # os.makedirs("%s/results" % save_path, exist_ok=True)
    # if os.path.isfile('%s/log.txt' % (save_path)):
    #     os.remove('%s/log.txt' % (save_path))

    hidden_states = []
    pool = torch.nn.AvgPool2d(2, 2)

    with torch.no_grad():
        preprocessor = PreprocessImage(K=K,
                                    old_width=Config.test_image_width,
                                    old_height=Config.test_image_height,
                                    new_width=Config.test_image_width,
                                    new_height=Config.test_image_height,
                                    distortion_crop=0,
                                    perform_crop=False)

        for i in range(2):
            reference_pose = poses[i]
            reference_image = load_image(image_filenames[i])

            reference_image = preprocessor.apply_rgb(image=reference_image, scale_rgb=scale_rgb, mean_rgb=mean_rgb, std_rgb=std_rgb)
            reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().to(device).unsqueeze(0)
            reference_feature_half, _, _, _ = feature_shrinker(*feature_extractor(reference_image_torch))
            keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half)


        for i in range(2, len(poses)):
            reference_pose = poses[i]
            reference_image = load_image(image_filenames[i])

            if param == 0:
                methods.append(0)

                reference_image = preprocessor.apply_rgb(image=reference_image, scale_rgb=scale_rgb, mean_rgb=mean_rgb, std_rgb=std_rgb)
                reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().to(device).unsqueeze(0)
                reference_pose_torch = torch.from_numpy(reference_pose).float().to(device).unsqueeze(0)

                full_K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().to(device).unsqueeze(0)

                half_K_torch = full_K_torch.clone().cuda()
                half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

                lstm_K_bottom = full_K_torch.clone().cuda()
                lstm_K_bottom[:, 0:2, :] = lstm_K_bottom[:, 0:2, :] / 32.0

                reference_feature_half, reference_feature_quarter, \
                reference_feature_one_eight, reference_feature_one_sixteen = feature_shrinker(*feature_extractor(reference_image_torch))

                measurement_poses_torch = []
                measurement_feature_halfs = []
                measurement_frames = keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
                for (measurement_pose, measurement_feature_half) in measurement_frames:
                    if measurement_feature_half.shape != reference_feature_half.shape:
                        measurement_feature_half = resize_torch(measurement_feature_half, reference_feature_half.shape, device)
                    measurement_feature_halfs.append(measurement_feature_half)
                    measurement_pose_torch = torch.from_numpy(measurement_pose).float().to(device).unsqueeze(0)
                    measurement_poses_torch.append(measurement_pose_torch)

                keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half)

                cost_volume = cost_volume_fusion(image1=reference_feature_half,
                                                image2s=measurement_feature_halfs,
                                                pose1=reference_pose_torch,
                                                pose2s=measurement_poses_torch,
                                                K=half_K_torch,
                                                warp_grid=warp_grid[0],
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

                if previous_depth[0] is not None and i - previous_idx[0] <= 10:
                # if previous_depth[0] is not None:
                    depth_estimation = get_non_differentiable_rectangle_depth_estimation(reference_pose_torch=reference_pose_torch,
                                                                                        measurement_pose_torch=previous_pose[0],
                                                                                        previous_depth_torch=previous_depth[0],
                                                                                        full_K_torch=full_K_torch,
                                                                                        half_K_torch=half_K_torch,
                                                                                        original_height=Config.test_image_height,
                                                                                        original_width=Config.test_image_width)
                    depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
                                                                    scale_factor=(1.0 / 16.0),
                                                                    mode="nearest")

                    ls = lstm_state[0]
                    pp = previous_pose[0]
                elif previous_depth[1] is not None:
                    new_shape = (1, 512, int(Config.test_image_height / 32.0), int(Config.test_image_width / 32.0))
                    ls = resize_torch(lstm_state[1][0].clone().cuda(), new_shape, device), resize_torch(lstm_state[1][1].clone().cuda(), new_shape, device)

                    pp = previous_pose[1].clone().cuda()
                    pp[:2] *= 2
                    pd = resize_torch(previous_depth[1], (1, 1, Config.test_image_height, Config.test_image_width), device)
                    depth_estimation = get_non_differentiable_rectangle_depth_estimation(reference_pose_torch=reference_pose_torch,
                                                                                        measurement_pose_torch=pp,
                                                                                        previous_depth_torch=pd,
                                                                                        full_K_torch=full_K_torch,
                                                                                        half_K_torch=half_K_torch,
                                                                                        original_height=Config.test_image_height,
                                                                                        original_width=Config.test_image_width)
                    depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
                                                                    scale_factor=(1.0 / 16.0),
                                                                    mode="nearest")

                else:
                    depth_estimation = torch.zeros(size=(1, 1, int(Config.test_image_height / 32.0), int(Config.test_image_width / 32.0))).to(device)

                    ls = lstm_state[0]
                    pp = previous_pose[0]

                lstm_state[0] = lstm_fusion(current_encoding=bottom,
                                        current_state=ls,
                                        previous_pose=pp,
                                        current_pose=reference_pose_torch,
                                        estimated_current_depth=depth_estimation,
                                        camera_matrix=lstm_K_bottom)

                hidden_states.append(pool(lstm_state[0][0]).cpu().numpy().squeeze())

                prediction = cost_volume_decoder(reference_image_torch, skip0, skip1, skip2, skip3, lstm_state[0][0])

                previous_depth[0] = prediction.view(1, 1, Config.test_image_height, Config.test_image_width)
                previous_pose[0] = reference_pose_torch
                previous_idx[0] = i

                prediction = prediction.cpu().numpy().squeeze()
            else:
                methods.append(1)

                reference_image = preprocessor.apply_rgb_resize(image=reference_image, scale_rgb=scale_rgb, mean_rgb=mean_rgb, std_rgb=std_rgb)
                reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().to(device).unsqueeze(0)
                reference_pose_torch = torch.from_numpy(reference_pose).float().to(device).unsqueeze(0)

                full_K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().to(device).unsqueeze(0)
                full_K_torch[:2] /= 2.0

                half_K_torch = full_K_torch.clone().cuda()
                half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

                lstm_K_bottom = full_K_torch.clone().cuda()
                lstm_K_bottom[:, 0:2, :] = lstm_K_bottom[:, 0:2, :] / 32.0

                reference_feature_half, reference_feature_quarter, \
                reference_feature_one_eight, reference_feature_one_sixteen = feature_shrinker(*feature_extractor(reference_image_torch))

                measurement_poses_torch = []
                measurement_feature_halfs = []
                measurement_frames = keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
                for (measurement_pose, measurement_feature_half) in measurement_frames:
                    if measurement_feature_half.shape != reference_feature_half.shape:
                        measurement_feature_half = resize_torch(measurement_feature_half, reference_feature_half.shape, device)
                    measurement_feature_halfs.append(measurement_feature_half)
                    measurement_pose_torch = torch.from_numpy(measurement_pose).float().to(device).unsqueeze(0)
                    measurement_poses_torch.append(measurement_pose_torch)

                keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half)

                cost_volume = cost_volume_fusion(image1=reference_feature_half,
                                                image2s=measurement_feature_halfs,
                                                pose1=reference_pose_torch,
                                                pose2s=measurement_poses_torch,
                                                K=half_K_torch,
                                                warp_grid=warp_grid[1],
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

                if previous_depth[1] is not None and i - previous_idx[0] >= 5:
                # if previous_depth[1] is not None:
                    depth_estimation = get_non_differentiable_rectangle_depth_estimation(reference_pose_torch=reference_pose_torch,
                                                                                        measurement_pose_torch=previous_pose[1],
                                                                                        previous_depth_torch=previous_depth[1],
                                                                                        full_K_torch=full_K_torch,
                                                                                        half_K_torch=half_K_torch,
                                                                                        original_height=Config.test_image_height // 2,
                                                                                        original_width=Config.test_image_width // 2)
                    depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
                                                                    scale_factor=(1.0 / 16.0),
                                                                    mode="nearest")

                    ls = lstm_state[1]
                    pp = previous_pose[1]
                elif previous_depth[0] is not None:
                    new_shape = (1, 512, int(Config.test_image_height / 64.0), int(Config.test_image_width / 64.0))
                    ls = resize_torch(lstm_state[0][0].clone().cuda(), new_shape, device), resize_torch(lstm_state[0][1].clone().cuda(), new_shape, device)

                    pp = previous_pose[0].clone().cuda()
                    pp[:2] /= 2
                    pd = resize_torch(previous_depth[0], (1, 1, Config.test_image_height // 2, Config.test_image_width // 2), device)
                    depth_estimation = get_non_differentiable_rectangle_depth_estimation(reference_pose_torch=reference_pose_torch,
                                                                                        measurement_pose_torch=pp,
                                                                                        previous_depth_torch=pd,
                                                                                        full_K_torch=full_K_torch,
                                                                                        half_K_torch=half_K_torch,
                                                                                        original_height=Config.test_image_height // 2,
                                                                                        original_width=Config.test_image_width // 2)
                    depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
                                                                    scale_factor=(1.0 / 16.0),
                                                                    mode="nearest")
                else:
                    depth_estimation = torch.zeros(size=(1, 1, int(Config.test_image_height / 64.0), int(Config.test_image_width / 64.0))).to(device)

                    ls = lstm_state[1]
                    pp = previous_pose[1]

                lstm_state[1] = lstm_fusion(current_encoding=bottom,
                                        current_state=ls,
                                        previous_pose=pp,
                                        current_pose=reference_pose_torch,
                                        estimated_current_depth=depth_estimation,
                                        camera_matrix=lstm_K_bottom)

                hidden_states.append(lstm_state[1][0].cpu().numpy().squeeze())

                prediction = cost_volume_decoder(reference_image_torch, skip0, skip1, skip2, skip3, lstm_state[1][0])

                previous_depth[1] = prediction.view(1, 1, Config.test_image_height // 2, Config.test_image_width // 2)
                previous_pose[1] = reference_pose_torch
                previous_idx[1] = i

                prediction = prediction.cpu().numpy().squeeze()
                prediction = cv2.resize(prediction, (Config.test_image_width, Config.test_image_height), interpolation=cv2.INTER_LINEAR)


            # cv2.imwrite('%s/results/%s' % (save_path, image_filenames[i].split("/")[-1]), (prediction * 10000).astype(np.uint16))

            reference_depth = cv2.imread(depth_filenames[i], -1).astype(float) / 10000.0
            mse, rmse = compute_errors(reference_depth, prediction)
            print('%04d %d: %.3f, %.3f' % (i, methods[-1], mse, rmse))
            # with open('%s/log.txt' % (save_path), 'a') as f:
            #     f.write('%04d %d: %.18e, %.18e\n' % (i, methods[-1], mse, rmse))
            mses.append(mse)
            rmses.append(rmse)

    return methods, mses, rmses, hidden_states

def prepare(test_dataset_name):
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

    scene = test_dataset_name
    scene_folder = Path('7scenes/%s' % scene)

    # scene = scene_folder.split("/")[-1]
    print("Predicting for scene:", scene)

    K = np.loadtxt(scene_folder / 'K.txt').astype(np.float32)
    poses = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
    image_filenames = sorted((scene_folder / 'images').files("*.png"))
    depth_filenames = sorted((scene_folder / 'depth').files("*.png"))

    warp_grid = [get_warp_grid_for_cost_volume_calculation(width=int(Config.test_image_width / 2),
                                                          height=int(Config.test_image_height / 2),
                                                          device=device),
                get_warp_grid_for_cost_volume_calculation(width=int(Config.test_image_width / 4),
                                                          height=int(Config.test_image_height / 4),
                                                          device=device)]

    return device, model, K, poses, image_filenames, depth_filenames, warp_grid


if __name__ == '__main__':
    dataset_names = {}
    # dataset_names["train"] = ["chess-seq-01", "chess-seq-02", "fire-seq-01", "fire-seq-02", "heads-seq-02", "office-seq-01", "office-seq-03", "pumpkin-seq-03", "pumpkin-seq-06", "redkitchen-seq-01", "redkitchen-seq-07", "stairs-seq-02", "stairs-seq-06"]
    dataset_names["train"] = ["redkitchen-seq-02", "redkitchen-seq-05", "redkitchen-seq-08", "chess-seq-04", "chess-seq-06", "office-seq-04", "office-seq-05", "office-seq-08", "pumpkin-seq-02", "pumpkin-seq-08", "stairs-seq-03", "stairs-seq-05"]
    dataset_names["test"] = ["chess-seq-03", "fire-seq-03", "fire-seq-04", "heads-seq-01", "office-seq-02", "pumpkin-seq-01", "redkitchen-seq-03", "stairs-seq-01"]

    params = [0, 100]
    # method_name = "random"

    for name in ["train", "test"]:
        rates, mse_errors, rmse_errors = [], [], []
        hidden_states = []
        for test_dataset_name in dataset_names[name]:
            methods, mses, rmses = [], [], []
            hs = []
            args = prepare(test_dataset_name)
            for param in params:
                print("\nPredicting for param: %d" % param)
                method, mse, rmse, hidden_state = predict(param, *args)
                methods.append(method)
                mses.append(mse)
                rmses.append(rmse)
                hs.append(hidden_state)

            if test_dataset_name == dataset_names[name][0]:
                for i in range(len(params)):
                    rates.append(methods[i])
                    mse_errors.append(mses[i])
                    rmse_errors.append(rmses[i])
                    hidden_states.append(hs[i])
            else:
                for i in range(len(params)):
                    rates[i].extend(methods[i])
                    mse_errors[i].extend(mses[i])
                    rmse_errors[i].extend(rmses[i])
                    hidden_states[i].extend(hs[i])

        rates = np.array(rates)
        rmse_errors = np.array(rmse_errors)
        mse_errors = np.array(mse_errors)

        np.savez_compressed('npzs/%sdata' % name, rates = rates, mses = mse_errors, rmses = rmse_errors, hidden_states = hidden_states)
        # np.savez_compressed('npzs/%s' % method_name, rates = np.mean(rates, axis=1) * 100, mses = np.mean(mse_errors, axis=1), rmses = np.mean(rmse_errors, axis=1))

        # print(method_name)
        print("params", params)
        print("rates", np.mean(rates, axis=1) * 100)
        print("MSE", np.mean(mse_errors, axis=1))
        print("RMSE", np.mean(rmse_errors, axis=1))
