#include "config.h"
#include "model.h"

void predict() {
    printf("Predicting with System: %s\n", system_name.c_str());
    printf("# of Measurement Frames: %d\n", test_n_measurement_frames);

    float warp_grid[3][warp_grid_width * warp_grid_height];
    get_warp_grid_for_cost_volume_calculation(warp_grid);

    // printvii(warp_grid, 3, warp_grid_width * warp_grid_height);

    printf("Predicting for scene:%s\n", scene.c_str());

    KeyframeBuffer keyframe_buffer;

    ifstream ifs;
    string file_buf;

    ifs = open_file(scene_folder + "/K.txt");
    float K[3][3];
    for (int i = 0; i < 3; i++) {
        getline(ifs, file_buf);
        istringstream iss(file_buf);
        string tmp;
        for (int j = 0; j < 3; j++) {
            iss >> tmp;
            K[i][j] = stof(tmp);
        }
    }

    // printvii(K, 3, 3);

    ifs = open_file(scene_folder + "/poses.txt");
    vector<float> tmp_poses;
    while (getline(ifs, file_buf)) {
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
    // printvii(poses[0], 4, 4);

    const string image_filedir = "/home/nhsmt1123/master-thesis/deep-video-mvs/sample-data/hololens-dataset/000/images/";
    const int len_image_filedir = image_filedir.length();
    string image_filenames[n_test_frames];
    for (int i = 0; i < n_test_frames; i++) {
        ostringstream sout;
        sout << setfill('0') << setw(5) << i+3;
        image_filenames[i] = image_filedir + sout.str() + ".png";
    }
    print1(image_filenames[0]);

    bool previous_exists = false;
    float previous_depth[test_image_height][test_image_width];
    float previous_pose[4][4];

    bool state_exists = false;
    float hidden_state[hyper_channels * 16][fe5_out_size(test_image_height)][fe5_out_size(test_image_width)];
    float cell_state[hyper_channels * 16][fe5_out_size(test_image_height)][fe5_out_size(test_image_width)];

    for (int f = 0; f < n_test_frames; f++) {
        float reference_pose[4][4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) reference_pose[i][j] = poses[f][i][j];
        float reference_image[org_image_height][org_image_width][3];
        load_image(image_filenames[f], reference_image);


        // POLL THE KEYFRAME BUFFER
        const int response = keyframe_buffer.try_new_keyframe(reference_pose, reference_image);
        cout << image_filenames[f].substr(len_image_filedir) << ": " << response << "\n";

        if (response == 0 || response == 2 || response == 4 || response == 5) continue;
        else if (response == 3) {
            previous_exists = false;
            state_exists = false;
            continue;
        }

        PreprocessImage preprocessor(K);

        float reference_image_torch[3][test_image_height][test_image_width];
        preprocessor.apply_rgb(reference_image, reference_image_torch);

        float reference_pose_torch[4][4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) reference_pose_torch[i][j] = reference_pose[i][j];

        float full_K_torch[3][3];
        preprocessor.get_updated_intrinsics(full_K_torch);

        float half_K_torch[3][3];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) half_K_torch[i][j] = full_K_torch[i][j] / 2.0;
        for (int j = 0; j < 3; j++) half_K_torch[2][j] = full_K_torch[2][j];

        float lstm_K_bottom[3][3];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) lstm_K_bottom[i][j] = full_K_torch[i][j] / 32.0;
        for (int j = 0; j < 3; j++) lstm_K_bottom[2][j] = full_K_torch[2][j];

        float measurement_poses[test_n_measurement_frames][4][4];
        float measurement_images[test_n_measurement_frames][org_image_height][org_image_width][3];
        const int n_measurement_frames = keyframe_buffer.get_best_measurement_frames(measurement_poses, measurement_images);

        float measurement_images_torch[test_n_measurement_frames][3][test_image_height][test_image_width];
        float measurement_poses_torch[test_n_measurement_frames][4][4];
        for (int m = 0; m < n_measurement_frames; m++) {
            float measurement_image[org_image_height][org_image_width][3];
            for (int i = 0; i < org_image_height; i++) for (int j = 0; j < org_image_width; j++) for (int k = 0; k < 3; k++)
                measurement_image[i][j][k] = measurement_images[m][i][j][k];
            preprocessor.apply_rgb(measurement_image, measurement_images_torch[m]);

            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) measurement_poses_torch[m][i][j] = measurement_poses[m][i][j];
        }


        FeatureExtractor<3, test_image_height, test_image_width> feature_extractor("params/0_feature_extractor");
        float layer1[fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)];
        float layer2[fe2_out_channels][fe2_out_size(test_image_height)][fe2_out_size(test_image_width)];
        float layer3[fe3_out_channels][fe3_out_size(test_image_height)][fe3_out_size(test_image_width)];
        float layer4[fe4_out_channels][fe4_out_size(test_image_height)][fe4_out_size(test_image_width)];
        float layer5[fe5_out_channels][fe5_out_size(test_image_height)][fe5_out_size(test_image_width)];

        FeatureShrinker<test_image_height, test_image_width> feature_shrinker("params/1_feature_pyramid");
        float measurement_feature_halfs[test_n_measurement_frames][fpn_output_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)];
        float measurement_feature_quarter[fpn_output_channels][fe2_out_size(test_image_height)][fe2_out_size(test_image_width)];
        float measurement_feature_one_eight[fpn_output_channels][fe3_out_size(test_image_height)][fe3_out_size(test_image_width)];
        float measurement_feature_one_sixteen[fpn_output_channels][fe4_out_size(test_image_height)][fe4_out_size(test_image_width)];

        for (int m = 0; m < n_measurement_frames; m++) {
            feature_extractor.forward(measurement_images_torch[m], layer1, layer2, layer3, layer4, layer5);
            feature_shrinker.forward(layer1, layer2, layer3, layer4, layer5, measurement_feature_halfs[m], measurement_feature_quarter, measurement_feature_one_eight, measurement_feature_one_sixteen);
        }
        if (f == 6) {
            // printvii(measurement_poses[m], 4, 4);
            // printviii(reference_image_torch, 3, 2, test_image_width);
            printviii(measurement_feature_halfs[0], 2, fe1_out_size(test_image_height), fe1_out_size(test_image_width));
        }
        break;

        float reference_feature_half[fpn_output_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)];
        float reference_feature_quarter[fpn_output_channels][fe2_out_size(test_image_height)][fe2_out_size(test_image_width)];
        float reference_feature_one_eight[fpn_output_channels][fe3_out_size(test_image_height)][fe3_out_size(test_image_width)];
        float reference_feature_one_sixteen[fpn_output_channels][fe4_out_size(test_image_height)][fe4_out_size(test_image_width)];
        feature_extractor.forward(reference_image_torch, layer1, layer2, layer3, layer4, layer5);
        feature_shrinker.forward(layer1, layer2, layer3, layer4, layer5, reference_feature_half,reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen);

        float cost_volume[n_depth_levels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)];
        cost_volume_fusion(reference_feature_half, measurement_feature_halfs, reference_pose_torch, measurement_poses_torch, half_K_torch, warp_grid, n_measurement_frames, cost_volume);

        float skip0[hyper_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)];
        float skip1[hyper_channels * 2][fe2_out_size(test_image_height)][fe2_out_size(test_image_width)];
        float skip2[hyper_channels * 4][fe3_out_size(test_image_height)][fe3_out_size(test_image_width)];
        float skip3[hyper_channels * 8][fe4_out_size(test_image_height)][fe4_out_size(test_image_width)];
        float bottom[hyper_channels * 16][fe5_out_size(test_image_height)][fe5_out_size(test_image_width)];
        CostVolumeEncoder<test_image_height, test_image_width> cost_volume_encoder("params/2_encoder");
        cost_volume_encoder.forward(reference_feature_half, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen, cost_volume,
                                    skip0, skip1, skip2, skip3, bottom);

        float depth_estimation[1][test_image_height / 32][test_image_width / 32];
        if (previous_exists) {
            float depth_hypothesis[1][test_image_height / 2][test_image_width / 2];
            get_non_differentiable_rectangle_depth_estimation(reference_pose_torch, previous_pose, previous_depth,
                                                              full_K_torch, half_K_torch,
                                                              depth_hypothesis);
            interpolate<1, test_image_height / 2, test_image_width / 2, test_image_height / 32, test_image_width / 32>(depth_hypothesis, depth_estimation);
        } else {
            for (int i = 0 ; i < test_image_height / 32; i++) for (int j = 0; j < test_image_width / 32; j++)
                depth_estimation[0][i][j] = 0;
        }

        LSTMFusion<fe5_out_size(test_image_height), fe5_out_size(test_image_width)> lstm_fusion("params/3_lstm_fusion");
        lstm_fusion.forward(bottom, previous_exists, previous_pose, reference_pose_torch, depth_estimation[0], lstm_K_bottom,
                            state_exists, hidden_state, cell_state);
        state_exists = true;

        float prediction[test_image_height][test_image_width];
        CostVolumeDecoder<test_image_height, test_image_width> cost_volume_decoder("params/4_decoder");
        cost_volume_decoder.forward(reference_image_torch, skip0, skip1, skip2, skip3, hidden_state, prediction);

        for (int i = 0 ; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++)
            previous_depth[i][j] = prediction[i][j];
        for (int i = 0 ; i < 4; i++) for (int j = 0; j < 4; j++)
            previous_pose[4][4] = reference_pose_torch[i][j];
        previous_exists = true;

        save_image("./results/" + image_filenames[f].substr(len_image_filedir), prediction);

        ofstream ofs("./results/" + image_filenames[f].substr(len_image_filedir, 5) + ".txt");
        for (int i = 0 ; i < test_image_height; i++) {
            for (int j = 0; j < test_image_width-1; j++)
                ofs << prediction[i][j] << " ";
            ofs << prediction[i][test_image_width-1] << "\n";
        }
        // if (f == 1) break;

    }
    keyframe_buffer.close();

}

int main() {
    predict();
    return 0;
}
