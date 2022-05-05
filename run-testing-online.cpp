#include "config.h"
#include "model.h"

void predict(const float reference_image[3 * test_image_height * test_image_width],
             const float reference_pose[4 * 4],
             const int n_measurement_frames,
             const float measurement_poses[test_n_measurement_frames][4 * 4],
             const float measurement_feature_halfs[test_n_measurement_frames][fpn_output_channels][height_2][width_2],
             const float half_K[3][3],
             const float warp_grid[3][width_2 * height_2],
             float hidden_state[hyper_channels * 16][height_32][width_32],
             float cell_state[hyper_channels * 16][height_32][width_32],
             float reference_feature_half[fpn_output_channels][height_2][width_2],
             float prediction[test_image_height][test_image_width]) {

    FeatureExtractor feature_extractor("params/0_feature_extractor");
    float layer1[channels_1][height_2][width_2];
    float layer2[channels_2][height_4][width_4];
    float layer3[channels_3][height_8][width_8];
    float layer4[channels_4][height_16][width_16];
    float layer5[channels_5][height_32][width_32];

    FeatureShrinker feature_shrinker("params/1_feature_pyramid");
    float reference_feature_quarter[fpn_output_channels][height_4][width_4];
    float reference_feature_one_eight[fpn_output_channels][height_8][width_8];
    float reference_feature_one_sixteen[fpn_output_channels][height_16][width_16];
    feature_extractor.forward(reference_image, layer1, layer2, layer3, layer4, layer5);
    feature_shrinker.forward(layer1, layer2, layer3, layer4, layer5, reference_feature_half, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen);

    if (n_measurement_frames == 0) return;

    float cost_volume[n_depth_levels][height_2][width_2];
    cost_volume_fusion(reference_feature_half, measurement_feature_halfs, reference_pose, measurement_poses, half_K, warp_grid, n_measurement_frames, cost_volume);

    float skip0[hyper_channels][height_2][width_2];
    float skip1[hyper_channels * 2][height_4][width_4];
    float skip2[hyper_channels * 4][height_8][width_8];
    float skip3[hyper_channels * 8][height_16][width_16];
    float bottom[hyper_channels * 16][height_32][width_32];
    CostVolumeEncoder cost_volume_encoder("params/2_encoder");
    cost_volume_encoder.forward(reference_feature_half, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen, cost_volume,
                                skip0, skip1, skip2, skip3, bottom);

    LSTMFusion lstm_fusion("params/3_lstm_fusion");
    lstm_fusion.forward(bottom, hidden_state, cell_state);

    CostVolumeDecoder cost_volume_decoder("params/4_decoder");
    cost_volume_decoder.forward(reference_image, skip0, skip1, skip2, skip3, hidden_state, prediction);

    // if (f == 6) {
    //     printvi(prediction[10], test_image_width);
    // }
    // break;
}


int main() {
    printf("Predicting with System: %s\n", system_name.c_str());
    printf("# of Measurement Frames: %d\n", test_n_measurement_frames);

    float warp_grid[3][width_2 * height_2];
    get_warp_grid_for_cost_volume_calculation(warp_grid);

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

    float full_K[3][3];
    get_updated_intrinsics(K, full_K);

    float half_K[3][3];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) half_K[i][j] = full_K[i][j] / 2.0;
    for (int j = 0; j < 3; j++) half_K[2][j] = full_K[2][j];

    float lstm_K_bottom[3][3];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) lstm_K_bottom[i][j] = full_K[i][j] / 32.0;
    for (int j = 0; j < 3; j++) lstm_K_bottom[2][j] = full_K[2][j];

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
    float poses[n_test_frames][4 * 4];
    int poses_idx = 0;
    for (int i = 0; i < n_test_frames; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                poses[i][j * 4 + k] = tmp_poses[poses_idx];
                poses_idx++;
            }
        }
    }
    // print1(n_poses);

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
    float previous_pose[4 * 4];

    bool state_exists = false;
    const int hid_channels = hyper_channels * 16;
    float hidden_state[hid_channels][height_32][width_32];
    float cell_state[hid_channels][height_32][width_32];

    for (int f = 0; f < n_test_frames; f++) {
        float reference_pose[4 * 4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) reference_pose[i * 4 + j] = poses[f][i * 4 + j];

        // POLL THE KEYFRAME BUFFER
        const int response = keyframe_buffer.try_new_keyframe(reference_pose);
        cout << image_filenames[f].substr(len_image_filedir) << ": " << response << "\n";

        if (response == 2 || response == 4 || response == 5) continue;
        else if (response == 3) {
            previous_exists = false;
            state_exists = false;
            continue;
        }

        float reference_image[3 * test_image_height * test_image_width];
        load_image(image_filenames[f], reference_image);

        float measurement_poses[test_n_measurement_frames][4 * 4];
        float measurement_feature_halfs[test_n_measurement_frames][fpn_output_channels][height_2][width_2];
        const int n_measurement_frames = keyframe_buffer.get_best_measurement_frames(reference_pose, measurement_poses, measurement_feature_halfs);

        // prepare depth_estimation
        float depth_estimation[1][height_32][width_32];
        if (previous_exists) {
            float depth_hypothesis[1][height_2][width_2];
            get_non_differentiable_rectangle_depth_estimation(reference_pose, previous_pose, previous_depth,
                                                                full_K, half_K,
                                                                depth_hypothesis);
            interpolate<1, height_2, width_2, height_32, width_32>(depth_hypothesis, depth_estimation);
        } else {
            for (int i = 0 ; i < height_32; i++) for (int j = 0; j < width_32; j++)
                depth_estimation[0][i][j] = 0;
        }

        // initialize ConvLSTM params if needed
        if (!state_exists) {
            for (int i = 0; i < hid_channels; i++) for (int j = 0; j < height_32; j++) for (int k = 0; k < width_32; k++)
                hidden_state[i][j][k] = 0;
            for (int i = 0; i < hid_channels; i++) for (int j = 0; j < height_32; j++) for (int k = 0; k < width_32; k++)
                cell_state[i][j][k] = 0;
        }

        if (previous_exists) {
            Matrix4f p_pose, c_pose;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) p_pose(i, j) = previous_pose[i * 4 + j];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c_pose(i, j) = reference_pose[i * 4 + j];

            Matrix4f transformation = p_pose.inverse() * c_pose;
            float trans[4][4];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) trans[i][j] = transformation(i, j);

            float tmp_hidden_state[hid_channels][height_32][width_32];
            warp_from_depth(hidden_state, depth_estimation[0], trans, lstm_K_bottom, tmp_hidden_state);

            for (int i = 0; i < hid_channels; i++) for (int j = 0; j < height_32; j++) for (int k = 0; k < width_32; k++)
                hidden_state[i][j][k] = (depth_estimation[0][j][k] <= 0.01) ? 0.0 : tmp_hidden_state[i][j][k];
        }

        float reference_feature_half[fpn_output_channels][height_2][width_2];
        float prediction[test_image_height][test_image_width];
        predict(reference_image, reference_pose, n_measurement_frames, measurement_poses, measurement_feature_halfs,
                half_K, warp_grid, hidden_state, cell_state, reference_feature_half, prediction);

        keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half);
        if (response == 0) continue;

        for (int i = 0 ; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++)
            previous_depth[i][j] = prediction[i][j];
        for (int i = 0 ; i < 4; i++) for (int j = 0; j < 4; j++)
            previous_pose[i * 4 + j] = reference_pose[i * 4 + j];
        previous_exists = true;

        state_exists = true;

        save_image("./results-hw/" + image_filenames[f].substr(len_image_filedir), prediction);

        ofstream ofs("./results-hw/" + image_filenames[f].substr(len_image_filedir, 5) + ".txt");
        for (int i = 0 ; i < test_image_height; i++) {
            for (int j = 0; j < test_image_width-1; j++)
                ofs << prediction[i][j] << " ";
            ofs << prediction[i][test_image_width-1] << "\n";
        }
    }

    keyframe_buffer.close();

    return 0;
}
