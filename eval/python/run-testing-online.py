import cv2
import numpy as np
import torch
from path import Path
import os

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
        return np.nan, np.nan

    differences = gt - pred
    squared_differences = np.square(differences)
    mse = np.mean(squared_differences)
    rmse = np.sqrt(mse)

    return mse, rmse


def predict(device, model, K, poses, image_filenames, depth_filenames, warp_grid):
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

    lstm_state = None
    previous_depth = None
    previous_pose = None

    mses = []
    rmses = []
    gts = []
    save_input = {}

    # save_path = "%s-%d/%s" % (method_name, param, test_dataset_name)
    # os.makedirs("%s/results" % save_path, exist_ok=True)
    # if os.path.isfile('%s/log.txt' % (save_path)):
    #     os.remove('%s/log.txt' % (save_path))

    with torch.no_grad():
        preprocessor = PreprocessImage(K=K,
                                    old_width=Config.org_image_width,
                                    old_height=Config.org_image_height,
                                    new_width=Config.test_image_width,
                                    new_height=Config.test_image_height,
                                    distortion_crop=0,
                                    perform_crop=False)

        for i in range(1):
            reference_pose = poses[i]
            reference_image = load_image(image_filenames[i])

            reference_image = preprocessor.apply_rgb(image=reference_image, scale_rgb=scale_rgb, mean_rgb=mean_rgb, std_rgb=std_rgb)
            reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().to(device).unsqueeze(0)
            reference_feature_half, _, _, _ = feature_shrinker(*feature_extractor(reference_image_torch))
            keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half)


        for i in range(1, len(poses)):
            reference_pose = poses[i]
            reference_image = load_image(image_filenames[i])

            reference_image = preprocessor.apply_rgb(image=reference_image, scale_rgb=scale_rgb, mean_rgb=mean_rgb, std_rgb=std_rgb)
            reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().to(device).unsqueeze(0)
            reference_pose_torch = torch.from_numpy(reference_pose).float().to(device).unsqueeze(0)

            full_K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().to(device).unsqueeze(0)

            half_K_torch = full_K_torch.clone().cuda()
            half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

            lstm_K_bottom = full_K_torch.clone().cuda()
            lstm_K_bottom[:, 0:2, :] = lstm_K_bottom[:, 0:2, :] / 32.0

            if "full_K" not in save_input:
                save_input["full_K"] = full_K_torch.cpu().detach().numpy().copy()
            if "half_K" not in save_input:
                save_input["half_K"] = half_K_torch.cpu().detach().numpy().copy()
            if "lstm_K" not in save_input:
                save_input["lstm_K"] = lstm_K_bottom.cpu().detach().numpy().copy()

            reference_feature_half, reference_feature_quarter, \
            reference_feature_one_eight, reference_feature_one_sixteen = feature_shrinker(*feature_extractor(reference_image_torch))

            if "reference_image" in save_input:
                save_input["reference_image"] = np.vstack([save_input["reference_image"], reference_image_torch.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_input["reference_image"] = reference_image_torch.cpu().detach().numpy().copy()[np.newaxis]
            if "reference_pose" in save_input:
                save_input["reference_pose"] = np.vstack([save_input["reference_pose"], reference_pose_torch.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_input["reference_pose"] = reference_pose_torch.cpu().detach().numpy().copy()[np.newaxis]

            measurement_poses_torch = []
            measurement_feature_halfs = []
            measurement_frames = keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
            for (measurement_pose, measurement_feature_half) in measurement_frames:
                measurement_feature_halfs.append(measurement_feature_half)
                measurement_pose_torch = torch.from_numpy(measurement_pose).float().to(device).unsqueeze(0)
                measurement_poses_torch.append(measurement_pose_torch)

            keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half)

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

            prediction = cost_volume_decoder(reference_image_torch, skip0, skip1, skip2, skip3, lstm_state[0])

            previous_depth = prediction.view(1, 1, Config.test_image_height, Config.test_image_width)
            previous_pose = reference_pose_torch

            prediction = prediction.cpu().numpy().squeeze()

            # cv2.imwrite('%s/results/%s' % (save_path, image_filenames[i].split("/")[-1]), (prediction * 10000).astype(np.uint16))
            reference_depth = cv2.imread(depth_filenames[i], -1).astype(float) / 10000.0
            reference_depth = cv2.resize(reference_depth, (Config.test_image_width, Config.test_image_height), interpolation=cv2.INTER_NEAREST)
            mse, rmse = compute_errors(reference_depth, prediction)
            print('%s: %.3f, %.3f' % (image_filenames[i].split('/')[-1][:-4], mse, rmse))
            # with open('%s/log.txt' % (save_path), 'a') as f:
            #     f.write('%04d: %.18e, %.18e\n' % (i, mse, rmse))
            mses.append(mse)
            rmses.append(rmse)
            gts.append(reference_depth)

    return mses, rmses, gts, save_input

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
    scene_folder = Path('/home/nhsmt1123/master-thesis/dvmvs-downsample2/7scenes/%s' % scene)

    # scene = scene_folder.split("/")[-1]
    print("Predicting for scene:", scene)

    K = np.loadtxt(scene_folder / 'K.txt').astype(np.float32)
    poses = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
    image_filenames = sorted((scene_folder / 'images').files("*.png"))
    depth_filenames = sorted((scene_folder / 'depth').files("*.png"))

    warp_grid = get_warp_grid_for_cost_volume_calculation(width=int(Config.test_image_width / 2),
                                                          height=int(Config.test_image_height / 2),
                                                          device=device)

    return device, model, K, poses, image_filenames, depth_filenames, warp_grid


if __name__ == '__main__':
    dataset_names = {}
    # dataset_names["train"] = ["chess-seq-01", "chess-seq-02", "fire-seq-01", "fire-seq-02", "heads-seq-02", "office-seq-01", "office-seq-03", "pumpkin-seq-03", "pumpkin-seq-06", "redkitchen-seq-01", "redkitchen-seq-07", "stairs-seq-02", "stairs-seq-06"]
    # dataset_names["train"] = ["redkitchen-seq-02", "redkitchen-seq-05", "redkitchen-seq-08", "chess-seq-04", "chess-seq-06", "office-seq-04", "office-seq-05", "office-seq-08", "pumpkin-seq-02", "pumpkin-seq-08", "stairs-seq-03", "stairs-seq-05"]
    # dataset_names["test"] = ["chess-seq-03", "fire-seq-03", "fire-seq-04", "heads-seq-01", "office-seq-02", "pumpkin-seq-01", "redkitchen-seq-03", "stairs-seq-01"]
    # test_dataset_names = ["chess-seq-01", "chess-seq-02", "fire-seq-01", "fire-seq-02", "heads-seq-02", "office-seq-01", "office-seq-03", "pumpkin-seq-03", "pumpkin-seq-06", "redkitchen-seq-01", "redkitchen-seq-07", "stairs-seq-02", "stairs-seq-06"]
    test_dataset_names = ["chess-seq-01", "chess-seq-02", "fire-seq-01", "fire-seq-02", "heads-seq-02", "office-seq-01", "office-seq-03", "redkitchen-seq-01", "redkitchen-seq-07"]

    # method_name = "random"
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    mses, rmses, gts = {}, {}, {}
    for test_dataset_name in test_dataset_names:
        args = prepare(test_dataset_name)
        print("Predicting: %s" % test_dataset_name)
        mse, rmse, gt, save_input = predict(*args)
        np.savez_compressed(base_dir / 'intrinsics/%s' % test_dataset_name, **save_input)
        print(test_dataset_name, "MSE", np.mean(mse))
        print(test_dataset_name, "RMSE", np.mean(rmse))
        mses[test_dataset_name] = mse
        rmses[test_dataset_name] = rmse
        gts[test_dataset_name] = gt

    np.savez_compressed(base_dir / 'mses', **mses)
    np.savez_compressed(base_dir / 'rmses', **rmses)
    np.savez_compressed(base_dir / 'gts', **gts)
    # np.savez_compressed('npzs/%s' % method_name, rates = np.mean(rates, axis=1) * 100, mses = np.mean(mse_errors, axis=1), rmses = np.mean(rmse_errors, axis=1))

