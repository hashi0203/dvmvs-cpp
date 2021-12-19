#include "config.h"
#include "model.h"

void predict() {
    printf("Predicting with System: %s\n", system_name.c_str());
    printf("# of Measurement Frames: %d\n", test_n_measurement_frames);

    // feature_extractor = FeatureExtractor();
    // feature_shrinker = FeatureShrinker();
    // cost_volume_encoder = CostVolumeEncoder();
    // lstm_fusion = LSTMFusion();
    // cost_volume_decoder = CostVolumeDecoder();

    float warp_grid[3][warp_grid_width * warp_grid_height];
    get_warp_grid_for_cost_volume_calculation(warp_grid);

    printvii(warp_grid, 3, warp_grid_width * warp_grid_height);

    const float min_depth = 0.25;
    const float max_depth = 20.0;
    const int n_depth_levels = 64;

    printf("Predicting for scene:%s\n", scene.c_str());

    KeyframeBuffer keyframe_buffer;

    ifstream ifs_K(scene_folder + "/K.txt");
    if (ifs_K.fail()) {
        cerr << "Failed to open file." << endl;
        return;
    }

    string file_buf;

    float K[3][3];
    for (int i = 0; i < 3; i++) {
        getline(ifs_K, file_buf);
        istringstream iss(file_buf);
        string tmp;
        for (int j = 0; j < 3; j++) {
            iss >> tmp;
            K[i][j] = stof(tmp);
        }
    }

    printvii(K, 3, 3);

    ifstream ifs_poses(scene_folder + "/poses.txt");
    if (ifs_poses.fail()) {
        cerr << "Failed to open file." << endl;
        return;
    }

    vector<float> tmp_poses;
    while (getline(ifs_poses, file_buf)) {
        istringstream iss(file_buf);
        string tmp;
        for (int i = 0; i < 16; i++) {
            iss >> tmp;
            tmp_poses.push_back(stof(tmp));
        }
    }

    // const int n_poses = tmp_poses.size() / 16;
    float poses[n_test_frames][4][4];
    int poses_idx = 0;
    for (int i = 0; i < n_test_frames; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                poses[i][j][k] = tmp_poses[poses_idx];
                poses_idx++;
            }
        }
    }
    // print1(n_poses);
    printvii(poses[0], 4, 4);

    string image_filenames[n_test_frames];
    for (int i = 0; i < n_test_frames; i++) {
        ostringstream sout;
        sout << setfill('0') << setw(5) << i+3;
        image_filenames[i] = "/home/nhsmt1123/master-thesis/deep-video-mvs/sample-data/hololens-dataset/000/images/" + sout.str() + ".png";
    }
    print1(image_filenames[0]);

    // lstm_state = None
    // previous_depth = None
    // previous_pose = None

    // predictions = []

    // reference_depths = None
    // depth_filenames = None

    for (int f = 0; f < n_test_frames; f++) {
        float reference_pose[4][4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) reference_pose[i][j] = poses[f][i][j];
        float reference_image[org_image_height][org_image_width][3];
        load_image(image_filenames[f], reference_image);

        // POLL THE KEYFRAME BUFFER
        const int response = keyframe_buffer.try_new_keyframe(reference_pose, reference_image);

        if (response == 0 || response == 2 || response == 4 || response == 5) continue;
        else if (response == 3) {
            // previous_depth = None
            // previous_pose = None
            // lstm_state = None
            continue;
        }

        PreprocessImage preprocessor(K);

        float reference_image_torch[3][test_image_height][test_image_width];
        preprocessor.apply_rgb(reference_image, reference_image_torch);

        float reference_pose_torch[4][4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) reference_pose_torch[i][j] = reference_pose[i][j];

        float full_K_torch[3][3];
        preprocessor.get_updated_intrinsics(full_K_torch);

        float half_K_torch[2][3];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) half_K_torch[i][j] = full_K_torch[i][j] / 2.0;

        float lstm_K_bottom[2][3];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) lstm_K_bottom[i][j] = full_K_torch[i][j] / 32.0;

        pair<float[4][4], float[org_image_height][org_image_width][3]> measurement_frames[test_n_measurement_frames];
        keyframe_buffer.get_best_measurement_frames(measurement_frames);

        float measurement_images_torch[test_n_measurement_frames][3][test_image_height][test_image_width];
        float measurement_poses_torch[test_n_measurement_frames][4][4];
        for (int m = 0; m < test_n_measurement_frames; m++) {
            float measurement_image[org_image_height][org_image_width][3];
            for (int i = 0; i < org_image_height; i++) for (int j = 0; j < org_image_width; j++) for (int k = 0; k < 3; k++)
                measurement_image[i][j][k] = measurement_frames[m].second[i][j][k];
            preprocessor.apply_rgb(measurement_image, measurement_images_torch[m]);

            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) measurement_poses_torch[m][i][j] = measurement_frames[m].first[i][j];
        }


        float measurement_feature_halfs[test_n_measurement_frames][fe1_out_channel][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)];

        float layer1[fe1_out_channel][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)];
        float layer2[fe2_out_channel][fe2_out_size(test_image_height)][fe2_out_size(test_image_width)];
        float layer3[fe3_out_channel][fe3_out_size(test_image_height)][fe3_out_size(test_image_width)];
        float layer4[fe4_out_channel][fe4_out_size(test_image_height)][fe4_out_size(test_image_width)];
        float layer5[fe5_out_channel][fe5_out_size(test_image_height)][fe5_out_size(test_image_width)];
        float measurement_feature_quarter[fe2_out_channel][fe2_out_size(test_image_height)][fe2_out_size(test_image_width)];
        float measurement_feature_one_eight[fe3_out_channel][fe3_out_size(test_image_height)][fe3_out_size(test_image_width)];
        float measurement_feature_one_sixteen[fe4_out_channel][fe4_out_size(test_image_height)][fe4_out_size(test_image_width)];

        FeatureExtractor<3, test_image_height, test_image_width> feature_extractor("params/0_feature_extractor");
        const int fpn_output_channels = 32;
        FeatureShrinker<test_image_height, test_image_width, fpn_output_channels> feature_shrinker("params/1_feature_pyramid");

        for (int m = 0; m < test_n_measurement_frames; m++) {
            feature_extractor.forward(measurement_images_torch[m], layer1, layer2, layer3, layer4, layer5);
            feature_shrinker.forward(layer1, layer2, layer3, layer4, layer5, measurement_feature_halfs[m], measurement_feature_quarter, measurement_feature_one_eight, measurement_feature_one_sixteen);
        }

    }

}

int main() {
    predict();
    return 0;
}

//             measurement_feature_halfs = []
//             for measurement_image_torch in measurement_images_torch:
//                 measurement_feature_half, _, _, _ = feature_shrinker(*feature_extractor(measurement_image_torch))
//                 measurement_feature_halfs.append(measurement_feature_half)

//             reference_feature_half, reference_feature_quarter, \
//             reference_feature_one_eight, reference_feature_one_sixteen = feature_shrinker(*feature_extractor(reference_image_torch))

//             cost_volume = cost_volume_fusion(image1=reference_feature_half,
//                                              image2s=measurement_feature_halfs,
//                                              pose1=reference_pose_torch,
//                                              pose2s=measurement_poses_torch,
//                                              K=half_K_torch,
//                                              warp_grid=warp_grid,
//                                              min_depth=min_depth,
//                                              max_depth=max_depth,
//                                              n_depth_levels=n_depth_levels,
//                                              device=device,
//                                              dot_product=True)

//             skip0, skip1, skip2, skip3, bottom = cost_volume_encoder(feature_half=reference_feature_half,
//                                                                      feature_quarter=reference_feature_quarter,
//                                                                      feature_one_eight=reference_feature_one_eight,
//                                                                      feature_one_sixteen=reference_feature_one_sixteen,
//                                                                      cost_volume=cost_volume)

//             if previous_depth is not None:
//                 depth_estimation = get_non_differentiable_rectangle_depth_estimation(reference_pose_torch=reference_pose_torch,
//                                                                                      measurement_pose_torch=previous_pose,
//                                                                                      previous_depth_torch=previous_depth,
//                                                                                      full_K_torch=full_K_torch,
//                                                                                      half_K_torch=half_K_torch,
//                                                                                      original_height=Config.test_image_height,
//                                                                                      original_width=Config.test_image_width)
//                 depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
//                                                                    scale_factor=(1.0 / 16.0),
//                                                                    mode="nearest")
//             else:
//                 depth_estimation = torch.zeros(size=(1, 1, int(Config.test_image_height / 32.0), int(Config.test_image_width / 32.0))).to(device)

//             lstm_state = lstm_fusion(current_encoding=bottom,
//                                      current_state=lstm_state,
//                                      previous_pose=previous_pose,
//                                      current_pose=reference_pose_torch,
//                                      estimated_current_depth=depth_estimation,
//                                      camera_matrix=lstm_K_bottom)

//             prediction, _, _, _, _ = cost_volume_decoder(reference_image_torch, skip0, skip1, skip2, skip3, lstm_state[0])
//             previous_depth = prediction.view(1, 1, Config.test_image_height, Config.test_image_width)
//             previous_pose = reference_pose_torch

//             inference_timer.record_end_time_and_elapsed_time()

//             prediction = prediction.cpu().numpy().squeeze()
//             predictions.append(prediction)

//             if Config.test_visualize:
//                 visualize_predictions(numpy_reference_image=reference_image,
//                                       numpy_measurement_image=measurement_image,
//                                       numpy_predicted_depth=prediction,
//                                       normalization_mean=mean_rgb,
//                                       normalization_std=std_rgb,
//                                       normalization_scale=scale_rgb,
//                                       depth_multiplier_for_visualization=5000)

//         inference_timer.print_statistics()

//         save_results(predictions=predictions,
//                      groundtruths=reference_depths,
//                      system_name=system_name,
//                      scene_name=scene,
//                      save_folder=".")