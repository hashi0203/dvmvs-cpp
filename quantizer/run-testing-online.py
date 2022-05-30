import cv2
import numpy as np
import torch
from path import Path
from tqdm import tqdm

from config import Config
from dataset_loader import PreprocessImage, load_image
from model_org import FeatureExtractor, FeatureShrinker, CostVolumeEncoder, LSTMFusion, CostVolumeDecoder
from keyframe_buffer2 import KeyframeBuffer
from utils import cost_volume_fusion, save_results, visualize_predictions, InferenceTimer, get_non_differentiable_rectangle_depth_estimation, \
    get_warp_grid_for_cost_volume_calculation


def predict(evaluate):
    dataset_name = Config.test_online_scene_path.split("/")[-2]
    system_name = "keyframe_{}_{}_{}_{}_dvmvs_fusionnet_online".format(dataset_name,
                                                                       Config.test_image_width,
                                                                       Config.test_image_height,
                                                                       Config.test_n_measurement_frames)

    base_dir = Path("/home/nhsmt1123/master-thesis/dvmvs-cpp-qt2/")

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

    predictions = []

    if evaluate:
        reference_depths = []
        depth_filenames = sorted((scene_folder / 'depth').files("*.png"))
    else:
        # if None the system will not be evaluated and errors will not be calculated
        reference_depths = None
        depth_filenames = None

    save_input = {}
    save_output = {}

    with torch.no_grad():
        for i in tqdm(range(0, 20)):
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

            if reference_depths is not None:
                reference_depth = cv2.imread(depth_filenames[i], -1).astype(float) / 1000.0
                reference_depth = preprocessor.apply_depth(reference_depth)
                reference_depths.append(reference_depth)

            reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().to(device).unsqueeze(0)
            reference_pose_torch = torch.from_numpy(reference_pose).float().to(device).unsqueeze(0)

            full_K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().to(device).unsqueeze(0)

            half_K_torch = full_K_torch.clone().cuda()
            half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

            lstm_K_bottom = full_K_torch.clone().cuda()
            lstm_K_bottom[:, 0:2, :] = lstm_K_bottom[:, 0:2, :] / 32.0

            # measurement_poses_torch = []
            # measurement_images_torch = []
            # measurement_frames = keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
            # for (measurement_pose, measurement_image) in measurement_frames:
            #     measurement_image = preprocessor.apply_rgb(image=measurement_image,
            #                                                scale_rgb=scale_rgb,
            #                                                mean_rgb=mean_rgb,
            #                                                std_rgb=std_rgb)
            #     measurement_image_torch = torch.from_numpy(np.transpose(measurement_image, (2, 0, 1))).float().to(device).unsqueeze(0)
            #     measurement_pose_torch = torch.from_numpy(measurement_pose).float().to(device).unsqueeze(0)
            #     measurement_images_torch.append(measurement_image_torch)
            #     measurement_poses_torch.append(measurement_pose_torch)

            inference_timer.record_start_time()

            # measurement_feature_halfs = []
            # for measurement_image_torch in measurement_images_torch:
            #     measurement_feature_half, _, _, _ = feature_shrinker(*feature_extractor(measurement_image_torch))
            #     measurement_feature_halfs.append(measurement_feature_half)

            if "input" in save_input:
                save_input["input"] = np.vstack([save_input["input"], reference_image_torch.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_input["input"] = reference_image_torch.cpu().detach().numpy().copy()[np.newaxis]

            layer1, layer2, layer3, layer4, layer5 = feature_extractor(reference_image_torch)

            if "layer1" in save_output:
                save_output["layer1"] = np.vstack([save_output["layer1"], layer1.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["layer1"] = layer1.cpu().detach().numpy().copy()[np.newaxis]
            if "layer2" in save_output:
                save_output["layer2"] = np.vstack([save_output["layer2"], layer2.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["layer2"] = layer2.cpu().detach().numpy().copy()[np.newaxis]
            if "layer3" in save_output:
                save_output["layer3"] = np.vstack([save_output["layer3"], layer3.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["layer3"] = layer3.cpu().detach().numpy().copy()[np.newaxis]
            if "layer4" in save_output:
                save_output["layer4"] = np.vstack([save_output["layer4"], layer4.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["layer4"] = layer4.cpu().detach().numpy().copy()[np.newaxis]
            if "layer5" in save_output:
                save_output["layer5"] = np.vstack([save_output["layer5"], layer5.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["layer5"] = layer5.cpu().detach().numpy().copy()[np.newaxis]

            reference_feature_half, reference_feature_quarter, \
            reference_feature_one_eight, reference_feature_one_sixteen = feature_shrinker(layer1, layer2, layer3, layer4, layer5)

            if "feature_half" in save_output:
                save_output["feature_half"] = np.vstack([save_output["feature_half"], reference_feature_half.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["feature_half"] = reference_feature_half.cpu().detach().numpy().copy()[np.newaxis]
            if "feature_quarter" in save_output:
                save_output["feature_quarter"] = np.vstack([save_output["feature_quarter"], reference_feature_quarter.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["feature_quarter"] = reference_feature_quarter.cpu().detach().numpy().copy()[np.newaxis]
            if "feature_one_eight" in save_output:
                save_output["feature_one_eight"] = np.vstack([save_output["feature_one_eight"], reference_feature_one_eight.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["feature_one_eight"] = reference_feature_one_eight.cpu().detach().numpy().copy()[np.newaxis]
            if "feature_one_sixteen" in save_output:
                save_output["feature_one_sixteen"] = np.vstack([save_output["feature_one_sixteen"], reference_feature_one_sixteen.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["feature_one_sixteen"] = reference_feature_one_sixteen.cpu().detach().numpy().copy()[np.newaxis]


            keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half)

            if response == 0:
                continue

            measurement_poses_torch = []
            measurement_feature_halfs = []
            measurement_frames = keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
            for (pose, feature_half) in measurement_frames:
                measurement_poses_torch.append(torch.from_numpy(pose).float().to(device).unsqueeze(0))
                measurement_feature_halfs.append(feature_half)

            mfhs = np.array([measurement_feature_half.cpu().detach().numpy().copy() for measurement_feature_half in measurement_feature_halfs]).reshape(-1, *reference_feature_half.shape)
            mfhs = np.concatenate([mfhs] + [np.zeros((1, *reference_feature_half.shape), dtype=mfhs.dtype) for _ in range(Config.test_n_measurement_frames - mfhs.shape[0])])
            if "measurement_features" in save_input:
                save_input["measurement_features"] = np.vstack([save_input["measurement_features"], mfhs[np.newaxis]])
            else:
                save_input["measurement_features"] = mfhs[np.newaxis]
            if "n_measurement_frames" in save_input:
                save_input["n_measurement_frames"].append(len(measurement_poses_torch))
            else:
                save_input["n_measurement_frames"] = [len(measurement_poses_torch)]
            if "half_K" in save_input:
                pass
            else:
                save_input["half_K"] = half_K_torch.cpu().detach().numpy().copy()
            if "pose1s" in save_input:
                save_input["pose1s"] = np.vstack([save_input["pose1s"], reference_pose_torch.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_input["pose1s"] = reference_pose_torch.cpu().detach().numpy().copy()[np.newaxis]
            pose2s = np.array([measurement_pose_torch.cpu().detach().numpy().copy() for measurement_pose_torch in measurement_poses_torch]).reshape(-1, *reference_pose_torch.shape)
            pose2s = np.concatenate([pose2s] + [np.zeros((1, *reference_pose_torch.shape), dtype=pose2s.dtype) for _ in range(Config.test_n_measurement_frames - pose2s.shape[0])])
            if "pose2ss" in save_input:
                save_input["pose2ss"] = np.vstack([save_input["pose2ss"], pose2s[np.newaxis]])
            else:
                save_input["pose2ss"] = pose2s[np.newaxis]

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

            if "cost_volume" in save_output:
                save_output["cost_volume"] = np.vstack([save_output["cost_volume"], cost_volume.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["cost_volume"] = cost_volume.cpu().detach().numpy().copy()[np.newaxis]


            skip0, skip1, skip2, skip3, bottom = cost_volume_encoder(features_half=reference_feature_half,
                                                                     features_quarter=reference_feature_quarter,
                                                                     features_one_eight=reference_feature_one_eight,
                                                                     features_one_sixteen=reference_feature_one_sixteen,
                                                                     cost_volume=cost_volume)

            if "skip0" in save_output:
                save_output["skip0"] = np.vstack([save_output["skip0"], skip0.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["skip0"] = skip0.cpu().detach().numpy().copy()[np.newaxis]
            if "skip1" in save_output:
                save_output["skip1"] = np.vstack([save_output["skip1"], skip1.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["skip1"] = skip1.cpu().detach().numpy().copy()[np.newaxis]
            if "skip2" in save_output:
                save_output["skip2"] = np.vstack([save_output["skip2"], skip2.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["skip2"] = skip2.cpu().detach().numpy().copy()[np.newaxis]
            if "skip3" in save_output:
                save_output["skip3"] = np.vstack([save_output["skip3"], skip3.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["skip3"] = skip3.cpu().detach().numpy().copy()[np.newaxis]
            if "bottom" in save_output:
                save_output["bottom"] = np.vstack([save_output["bottom"], bottom.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["bottom"] = bottom.cpu().detach().numpy().copy()[np.newaxis]


            if "full_K" in save_input:
                pass
            else:
                save_input["full_K"] = full_K_torch.cpu().detach().numpy().copy()

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

            if "hidden_state" in save_input:
                save_input["hidden_state"] = np.vstack([save_input["hidden_state"], lstm_state[0].cpu().detach().numpy().copy()[np.newaxis]])
            else:
                if lstm_state is None:
                    save_input["hidden_state"] = lstm_fusion.lstm_cell.init_hidden(batch_size=bottom.size()[0], image_size=bottom.size()[2:])[0].cpu().detach().numpy().copy()[np.newaxis]
                else:
                    save_input["hidden_state"] = lstm_state[0].cpu().detach().numpy().copy()[np.newaxis]
            if "cell_state" in save_input:
                save_input["cell_state"] = np.vstack([save_input["cell_state"], lstm_state[1].cpu().detach().numpy().copy()[np.newaxis]])
            else:
                if lstm_state is None:
                    save_input["cell_state"] = lstm_fusion.lstm_cell.init_hidden(batch_size=bottom.size()[0], image_size=bottom.size()[2:])[1].cpu().detach().numpy().copy()[np.newaxis]
                else:
                    save_input["cell_state"] = lstm_state[1].cpu().detach().numpy().copy()[np.newaxis]
            if "lstm_K" in save_input:
                pass
            else:
                save_input["lstm_K"] = lstm_K_bottom.cpu().detach().numpy().copy()

            lstm_state = lstm_fusion(current_encoding=bottom,
                                     current_state=lstm_state,
                                     previous_pose=previous_pose,
                                     current_pose=reference_pose_torch,
                                     estimated_current_depth=depth_estimation,
                                     camera_matrix=lstm_K_bottom)

            if "hidden_state" in save_output:
                save_output["hidden_state"] = np.vstack([save_output["hidden_state"], lstm_state[0].cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["hidden_state"] = lstm_state[0].cpu().detach().numpy().copy()[np.newaxis]
            if "cell_state" in save_output:
                save_output["cell_state"] = np.vstack([save_output["cell_state"], lstm_state[1].cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["cell_state"] = lstm_state[1].cpu().detach().numpy().copy()[np.newaxis]


            prediction, _, _, _, _, depth_org = cost_volume_decoder(reference_image_torch, skip0, skip1, skip2, skip3, lstm_state[0])
            previous_depth = prediction.view(1, 1, Config.test_image_height, Config.test_image_width)
            previous_pose = reference_pose_torch

            if "depth_org" in save_output:
                save_output["depth_org"] = np.vstack([save_output["depth_org"], depth_org.cpu().detach().numpy().copy()[np.newaxis]])
            else:
                save_output["depth_org"] = depth_org.cpu().detach().numpy().copy()[np.newaxis]

            inference_timer.record_end_time_and_elapsed_time()

            prediction = prediction.cpu().numpy().squeeze()
            predictions.append(prediction)

            save_path = Path("/home/nhsmt1123/master-thesis/dvmvs-cpp/results-org")
            cv2.imwrite(save_path / image_filenames[i].split("/")[-1], (prediction * 25).astype(np.uint8))
            with open(save_path / image_filenames[i].split("/")[-1][:-4] + ".txt", 'w') as f:
                for p in prediction:
                    f.write(' '.join(map(str, p)) + '\n')

            # if Config.test_visualize:
            #     visualize_predictions(numpy_reference_image=reference_image,
            #                           numpy_measurement_image=measurement_image,
            #                           numpy_predicted_depth=prediction,
            #                           normalization_mean=mean_rgb,
            #                           normalization_std=std_rgb,
            #                           normalization_scale=scale_rgb,
            #                           depth_multiplier_for_visualization=5000)

        inference_timer.print_statistics()

        # save_results(predictions=predictions,
        #              groundtruths=reference_depths,
        #              system_name=system_name,
        #              scene_name=scene,
        #              save_folder=".")

    np.savez_compressed(base_dir / "params" / "inputs", **save_input)
    np.savez_compressed(base_dir / "params" / "outputs", **save_output)

if __name__ == '__main__':
    predict(evaluate=True)
