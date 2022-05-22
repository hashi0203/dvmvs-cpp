#include "config.h"
#include "functional.h"
#include "conv.h"
#include "activation.h"
#include "layers.h"
#include "mnasnet.h"
#include "model.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
using namespace Eigen;

// qwint* params = new qwint[2725512 + 62272 + 8990848 + 18874368 + 4066277];
// int start_idx[n_files + 1];
// int param_cnt;
// int shifts[n_files];
// int offset_cnt;
// int actshifts[n_acts];
// int act_cnt;

// float* params_f = new float[2725512 + 62272 + 8990848 + 18874368 + 4066277];


qwint* weights = new qwint[n_weights];
int w_idx[w_files];
int w_shifts[w_files];
int w_cnt;

qbint* biases = new qbint[n_biases];
int b_idx[b_files];
int b_shifts[b_files];
int b_cnt;

qsint* scales = new qsint[n_scales];
int s_idx[s_files];
int s_shifts[s_files];
int s_cnt;

int a_shifts[a_files];
int a_cnt;


const string save_dir = "./results-qt/";

void set_idx(string filename, const int n_files, int* start_idx) {
    int n_params[n_files];
    ifstream ifs("params/" + filename);
    ifs.read((char*) n_params, sizeof(int) * n_files);
    ifs.close();

    start_idx[0] = 0;
    for (int i = 0; i < n_files - 1; i++)
        start_idx[i+1] = start_idx[i] + n_params[i];
}


template<class T>
void set_param(string filename, const int n_params, T* params) {
    ifstream ifs("params/" + filename);
    ifs.read((char*) params, sizeof(T) * n_params);
    ifs.close();
}


void read_params() {
    ifstream ifs;

    set_idx("n_weights", w_files, w_idx);
    set_param<qwint>("weights_quantized", n_weights, weights);
    set_param<int>("weight_shifts", n_weights, w_shifts);

    set_idx("n_biases", b_files, b_idx);
    set_param<qbint>("biases_quantized", n_biases, biases);
    set_param<int>("bias_shifts", n_biases, b_shifts);

    set_idx("n_scales", s_files, s_idx);
    set_param<qsint>("scales_quantized", n_scales, scales);
    set_param<int>("scale_shifts", n_scales, s_shifts);

    set_param<int>("act_shifts", n_acts, a_shifts);


    // int n_params[n_files];
    // ifs.open("params/values_quantized");
    // ifs.read((char*) n_params, sizeof(int) * n_files);
    // ifs.close();

    // start_idx[0] = 0;
    // for (int i = 0; i < n_files; i++)
    //     start_idx[i+1] = start_idx[i] + n_params[i];

    // ifs.open("params/params_quantized");
    // ifs.read((char*) params, sizeof(qwint) * start_idx[n_files]);
    // ifs.close();

    // ifs.open("params/params");
    // ifs.read((char*) params_f, sizeof(float) * start_idx[n_files]);
    // ifs.close();

    // ifs.open("params/shifts_quantized");
    // ifs.read((char*) shifts, sizeof(int) * n_files);
    // ifs.close();

    // ifs.open("params/actshifts_quantized");
    // ifs.read((char*) actshifts, sizeof(int) * n_acts);
    // ifs.close();

    // for (int idx = 0; idx < n_acts; idx++)
    //     actshifts[idx] += 4;
}


void predict(const qaint reference_image[3 * test_image_height * test_image_width],
             const int n_measurement_frames,
             const qaint measurement_feature_halfs[test_n_measurement_frames * fpn_output_channels * height_2 * width_2],
             const float* warpings,
             qaint reference_feature_half[fpn_output_channels * height_2 * width_2],
             float hidden_state[hid_channels * height_32 * width_32],
             float cell_state[hid_channels * height_32 * width_32],
             float prediction[test_image_height * test_image_width],
             const string filename) {

    // param_cnt = 0;
    // offset_cnt = 0;
    // act_cnt = 0;

    w_cnt = 0;
    b_cnt = 0;
    s_cnt = 0;
    a_cnt = 0;

    // int ashift;
    // float reference_image_float[3 * test_image_height * test_image_width];
    // ashift = actshifts[0];
    // for (int idx = 0; idx < 3 * test_image_height * test_image_width; idx++)
    //     reference_image_float[idx] = reference_image[idx] / (float) (1 << ashift);

    qaint layer1[channels_1 * height_2 * width_2];
    qaint layer2[channels_2 * height_4 * width_4];
    qaint layer3[channels_3 * height_8 * width_8];
    qaint layer4[channels_4 * height_16 * width_16];
    qaint layer5[channels_5 * height_32 * width_32];
    FeatureExtractor(reference_image, layer1, layer2, layer3, layer4, layer5);
    ofstream ofs2(save_dir + "layer2-" + filename + ".txt");
    for (int idx = 0; idx < channels_2 * height_4 * width_4; idx++)
        ofs2 << layer2[idx] / (float) (1 << a_shifts[14]) << "\n";
    ofs2.close();

    ofstream ofs5(save_dir + "layer5-" + filename + ".txt");
    // ofstream ofs5("layer5.txt", ios::out|ios::binary|ios::trunc);
    // for (int idx = 0; idx < channels_1 * height_2 * width_2; idx++)
    // for (int idx = 0; idx < channels_2 * height_4 * width_4; idx++)
    for (int idx = 0; idx < channels_5 * height_32 * width_32; idx++)
        ofs5 << layer5[idx] / (float) (1 << a_shifts[a_cnt]) << "\n";
        // ofs5 << layer2[idx] / (float) (1 << actshifts[24]) << "\n";
        // ofs5 << layer1[idx] / (float) (1 << actshifts[6]) << "\n";
        // ofs5.write((char*) &layer5[idx], sizeof(float));
    ofs5.close();

    // float reference_feature_half_float[fpn_output_channels * height_2 * width_2];

    // float reference_feature_quarter[fpn_output_channels * height_4 * width_4];
    // float reference_feature_one_eight[fpn_output_channels * height_8 * width_8];
    // float reference_feature_one_sixteen[fpn_output_channels * height_16 * width_16];
    // FeatureShrinker(layer1, layer2, layer3, layer4, layer5, reference_feature_half_float, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen);

    // ashift = actshifts[6];
    // for (int idx = 0; idx < fpn_output_channels * height_2 * width_2; idx++)
    //     reference_feature_half[idx] = reference_feature_half_float[idx] * (1 << ashift);

    // // ofstream ofsf("feature_half.txt", ios::out|ios::binary|ios::trunc);
    // // for (int idx = 0; idx < fpn_output_channels * height_2 * width_2; idx++)
    // //     // ofsf << reference_feature_half[idx] << "\n";
    // //     ofsf.write((char*) &reference_feature_half[idx], sizeof(float));
    // // ofsf.close();

    // if (n_measurement_frames == 0) return;

    // float measurement_feature_halfs_float[test_n_measurement_frames * fpn_output_channels * height_2 * width_2];
    // ashift = actshifts[6];
    // for (int idx = 0; idx < test_n_measurement_frames * fpn_output_channels * height_2 * width_2; idx++)
    //     measurement_feature_halfs_float[idx] = measurement_feature_halfs[idx] / (float) (1 << ashift);

    // float cost_volume[n_depth_levels * height_2 * width_2];
    // cost_volume_fusion(reference_feature_half_float, n_measurement_frames, measurement_feature_halfs_float, warpings, cost_volume);

    // // // ofstream ofsc("cost_volume.txt");
    // // ofstream ofsc("cost_volume.txt", ios::out|ios::binary|ios::trunc);
    // // for (int idx = 0; idx < n_depth_levels * height_2 * width_2; idx++)
    // //     // ofsc << cost_volume[idx] << "\n";
    // //     ofsc.write((char*) &cost_volume[idx], sizeof(float));
    // // ofsc.close();

    // float skip0[hyper_channels * height_2 * width_2];
    // float skip1[(hyper_channels * 2) * height_4 * width_4];
    // float skip2[(hyper_channels * 4) * height_8 * width_8];
    // float skip3[(hyper_channels * 8) * height_16 * width_16];
    // float bottom[(hyper_channels * 16) * height_32 * width_32];
    // CostVolumeEncoder(reference_feature_half_float, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen, cost_volume,
    //                   skip0, skip1, skip2, skip3, bottom);

    // // // ofstream ofsb("bottom.txt");
    // // ofstream ofsb("bottom.txt", ios::out|ios::binary|ios::trunc);
    // // for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
    // //     // ofsb << bottom[idx] << "\n";
    // //     ofsb.write((char*) &bottom[idx], sizeof(float));
    // // ofsb.close();

    // LSTMFusion(bottom, hidden_state, cell_state);

    // // // ofstream ofsh("hidden_state.txt");
    // // ofstream ofsh("hidden_state.txt", ios::out|ios::binary|ios::trunc);
    // // for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
    // //     // ofsh << hidden_state[idx] << "\n";
    // //     ofsh.write((char*) &hidden_state[idx], sizeof(float));
    // // ofsh.close();

    // CostVolumeDecoder(reference_image_float, skip0, skip1, skip2, skip3, hidden_state, prediction);
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

    ifs.open(scene_folder + "/K.txt");
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
    ifs.close();

    float full_K[3][3];
    get_updated_intrinsics(K, full_K);

    float half_K[3][3];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) half_K[i][j] = full_K[i][j] / 2.0;
    for (int j = 0; j < 3; j++) half_K[2][j] = full_K[2][j];

    float lstm_K_bottom[3][3];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) lstm_K_bottom[i][j] = full_K[i][j] / 32.0;
    for (int j = 0; j < 3; j++) lstm_K_bottom[2][j] = full_K[2][j];

    ifs.open(scene_folder + "/poses.txt");
    vector<float> tmp_poses;
    while (getline(ifs, file_buf)) {
        istringstream iss(file_buf);
        string tmp;
        for (int i = 0; i < 16; i++) {
            iss >> tmp;
            tmp_poses.push_back(stof(tmp));
        }
    }
    ifs.close();

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

    // read params
    read_params();

    bool previous_exists = false;
    float previous_depth[test_image_height][test_image_width];
    float previous_pose[4 * 4];

    bool state_exists = false;
    float hidden_state[hid_channels * height_32 * width_32];
    float cell_state[hid_channels * height_32 * width_32];

    for (int f = 6; f < n_test_frames; f++) {
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

        float reference_image_float[3 * test_image_height * test_image_width];
        load_image(image_filenames[f], reference_image_float);
        qaint reference_image[3 * test_image_height * test_image_width];
        const int ashift = a_shifts[0];
        for (int idx = 0; idx < 3 * test_image_height * test_image_width; idx++)
            reference_image[idx] = reference_image_float[idx] * (1 << ashift);

        float measurement_poses[test_n_measurement_frames * 4 * 4];
        qaint measurement_feature_halfs[test_n_measurement_frames * fpn_output_channels * height_2 * width_2];
        const int n_measurement_frames = keyframe_buffer.get_best_measurement_frames(reference_pose, measurement_poses, measurement_feature_halfs);

        // prepare for cost volume fusion
        float* warpings = new float[n_measurement_frames * n_depth_levels * height_2 * width_2 * 2];

        for (int m = 0; m < n_measurement_frames; m++) {
            Matrix4f pose1, pose2;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) pose1(i, j) = reference_pose[i * 4 + j];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) pose2(i, j) = measurement_poses[(m * 4 + i) * 4 + j];

            Matrix4f extrinsic2 = pose2.inverse() * pose1;
            Matrix3f R = extrinsic2.block(0, 0, 3, 3);
            Vector3f t = extrinsic2.block(0, 3, 3, 1);

            Matrix3f K;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) K(i, j) = half_K[i][j];
            MatrixXf wg(3, width_2 * height_2);
            for (int i = 0; i < 3; i++) for (int j = 0; j < width_2 * height_2; j++) wg(i, j) = warp_grid[i][j];

            Vector3f _Kt = K * t;
            Matrix3f K_R_Kinv = K * R * K.inverse();
            MatrixXf K_R_Kinv_UV(3, width_2 * height_2);
            K_R_Kinv_UV = K_R_Kinv * wg;

            MatrixXf Kt(3, width_2 * height_2);
            for (int i = 0; i < width_2 * height_2; i++) Kt.block(0, i, 3, 1) = _Kt;

            for (int depth_i = 0; depth_i < n_depth_levels; depth_i++) {
                const float this_depth = 1.0 / (inverse_depth_base + depth_i * inverse_depth_step);

                MatrixXf _warping(width_2 * height_2, 3);
                _warping = (K_R_Kinv_UV + (Kt / this_depth)).transpose();

                MatrixXf _warping0(width_2 * height_2, 2);
                VectorXf _warping1(width_2 * height_2);
                _warping0 = _warping.block(0, 0, width_2 * height_2, 2);
                _warping1 = _warping.block(0, 2, width_2 * height_2, 1).array() + 1e-8f;

                _warping0.block(0, 0, width_2 * height_2, 1).array() /= _warping1.array();
                _warping0.block(0, 0, width_2 * height_2, 1).array() -= width_normalizer;
                _warping0.block(0, 0, width_2 * height_2, 1).array() /= width_normalizer;

                _warping0.block(0, 1, width_2 * height_2, 1).array() /= _warping1.array();
                _warping0.block(0, 1, width_2 * height_2, 1).array() -= height_normalizer;
                _warping0.block(0, 1, width_2 * height_2, 1).array() /= height_normalizer;

                for (int idx = 0; idx < height_2 * width_2; idx++) for (int k = 0; k < 2; k++)
                    warpings[((m * n_depth_levels + depth_i) * (height_2 * width_2) + idx) * 2 + k] = _warping0(idx, k);
            }
        }

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
            for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
                hidden_state[idx] = 0;
            for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
                cell_state[idx] = 0;
        }

        if (previous_exists) {
            Matrix4f p_pose, c_pose;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) p_pose(i, j) = previous_pose[i * 4 + j];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c_pose(i, j) = reference_pose[i * 4 + j];

            Matrix4f transformation = p_pose.inverse() * c_pose;
            float trans[4][4];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) trans[i][j] = transformation(i, j);

            float in_hidden_state[hid_channels][height_32][width_32];
            for (int i = 0; i < hid_channels; i++) for (int j = 0; j < height_32; j++) for (int k = 0; k < width_32; k++)
                in_hidden_state[i][j][k] = hidden_state[(i * height_32 + j) * width_32 + k];
            float out_hidden_state[hid_channels][height_32][width_32];
            warp_from_depth(in_hidden_state, depth_estimation[0], trans, lstm_K_bottom, out_hidden_state);

            for (int i = 0; i < hid_channels; i++) for (int j = 0; j < height_32; j++) for (int k = 0; k < width_32; k++)
                hidden_state[(i * height_32 + j) * width_32 + k] = (depth_estimation[0][j][k] <= 0.01) ? 0.0 : out_hidden_state[i][j][k];
        }

        qaint reference_feature_half[fpn_output_channels * height_2 * width_2];
        float prediction[test_image_height * test_image_width];
        predict(reference_image, n_measurement_frames, measurement_feature_halfs,
                warpings, reference_feature_half, hidden_state, cell_state, prediction, image_filenames[f].substr(len_image_filedir, 5));
        delete[] warpings;
        break;

        keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half);
        if (response == 0) continue;

        for (int i = 0 ; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++)
            previous_depth[i][j] = prediction[i * test_image_width + j];
        for (int i = 0 ; i < 4; i++) for (int j = 0; j < 4; j++)
            previous_pose[i * 4 + j] = reference_pose[i * 4 + j];
        previous_exists = true;

        state_exists = true;

        save_image(save_dir + image_filenames[f].substr(len_image_filedir), previous_depth);

        ofstream ofs(save_dir + image_filenames[f].substr(len_image_filedir, 5) + ".txt");
        for (int i = 0 ; i < test_image_height; i++) {
            for (int j = 0; j < test_image_width-1; j++)
                ofs << previous_depth[i][j] << " ";
            ofs << previous_depth[i][test_image_width-1] << "\n";
        }
    }

    keyframe_buffer.close();

    delete[] weights;
    delete[] biases;
    delete[] scales;
    // delete[] params;
    // delete[] params_f;

    return 0;
}
