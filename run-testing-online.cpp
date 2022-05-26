#include "config.h"

template<class T>
void save_layer(string save_dir, string layer_name, string filename, const T* layer, const int layer_size, const int shift, string mode="txt") {
    if (mode == "txt") {
        ofstream ofs(save_dir + layer_name + "-" + filename + ".txt");
        for (int idx = 0; idx < layer_size; idx++)
            ofs << layer[idx] / (float) (1 << shift) << "\n";
        ofs.close();
    } else if (mode == "bin") {
        ofstream ofs(save_dir + layer_name + "-" + filename, ios::out|ios::binary|ios::trunc);
        for (int idx = 0; idx < layer_size; idx++)
            ofs.write((char*) &layer[idx], sizeof(T));
        ofs.close();
    } else {
        print2("unexpected mode:", mode);
    }
}

#include "functional.h"
#include "activation.h"
#include "conv.h"
#include "layers.h"
#include "mnasnet.h"
#include "model.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
using namespace Eigen;

int conv_cnt;
int bn_cnt;
int add_cnt;
int other_cnt;
int act_cnt;

qwint* weights = new qwint[n_weights];
int w_idx[n_convs];
int w_shifts[n_convs];

qbint* biases = new qbint[n_biases];
int b_idx[n_convs];
int b_shifts[n_convs];

qsint* scales = new qsint[n_scales];
int s_idx[n_bns];
int s_shifts[n_bns];

int cin_shifts[n_convs];
int cout_shifts[n_convs];
int ain1_shifts[n_adds];
int ain2_shifts[n_adds];
int aout_shifts[n_adds];
int oin_shifts[n_others];
int oout_shifts[n_others];

int ln_cnt;


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
    set_idx("n_weights", n_convs, w_idx);
    set_param<qwint>("weights_quantized", n_weights, weights);
    set_param<int>("weight_shifts", n_convs, w_shifts);

    set_idx("n_biases", n_convs, b_idx);
    set_param<qbint>("biases_quantized", n_biases, biases);
    set_param<int>("bias_shifts", n_convs, b_shifts);

    set_idx("n_scales", n_bns, s_idx);
    set_param<qsint>("scales_quantized", n_scales, scales);
    set_param<int>("scale_shifts", n_bns, s_shifts);

    set_param<int>("cin_shifts", n_convs, cin_shifts);
    set_param<int>("cout_shifts", n_convs, cout_shifts);
    set_param<int>("ain1_shifts", n_adds, ain1_shifts);
    set_param<int>("ain2_shifts", n_adds, ain2_shifts);
    set_param<int>("aout_shifts", n_adds, aout_shifts);
    set_param<int>("oin_shifts", n_others, oin_shifts);
    set_param<int>("oout_shifts", n_others, oout_shifts);

    constexpr int irregulars[4] = {64, 72, 78, 93}; // 応急処置
    for (int idx : irregulars) cin_shifts[idx]--;
}


void predict(const qaint reference_image[3 * test_image_height * test_image_width],
             const int n_measurement_frames,
             const qaint measurement_feature_halfs[test_n_measurement_frames * fpn_output_channels * height_2 * width_2],
             const float* warpings,
             qaint reference_feature_half[fpn_output_channels * height_2 * width_2],
             qaint hidden_state[hid_channels * height_32 * width_32],
             qaint cell_state[hid_channels * height_32 * width_32],
             qaint depth_full[test_image_height * test_image_width],
             const string filename) {

    conv_cnt = 0;
    bn_cnt = 0;
    add_cnt = 0;
    other_cnt = 0;
    act_cnt = 0;
    ln_cnt = 0;

    qaint layer1[channels_1 * height_2 * width_2];
    qaint layer2[channels_2 * height_4 * width_4];
    qaint layer3[channels_3 * height_8 * width_8];
    qaint layer4[channels_4 * height_16 * width_16];
    qaint layer5[channels_5 * height_32 * width_32];
    int act_out_layer1;
    int act_out_layer2;
    int act_out_layer3;
    int act_out_layer4;
    int act_out_layer5;
    FeatureExtractor(reference_image, layer1, layer2, layer3, layer4, layer5,
                     act_cnt++, act_out_layer1, act_out_layer2, act_out_layer3, act_out_layer4, act_out_layer5);

    // save_layer<qaint>(save_dir, "layer1", filename, layer1, channels_1 * height_2 * width_2, cout_shifts[3-1]);
    // save_layer<qaint>(save_dir, "layer2", filename, layer2, channels_2 * height_4 * width_4, cout_shifts[12-1]);
    // save_layer<qaint>(save_dir, "layer5", filename, layer5, channels_5 * height_32 * width_32, cout_shifts[conv_cnt-1]);

    // qaint reference_feature_quarter[fpn_output_channels * height_4 * width_4];
    // qaint reference_feature_one_eight[fpn_output_channels * height_8 * width_8];
    // qaint reference_feature_one_sixteen[fpn_output_channels * height_16 * width_16];
    // FeatureShrinker(layer1, layer2, layer3, layer4, layer5, reference_feature_half, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen);

    // save_layer<qaint>(save_dir, "feature_one_sixteen", filename, reference_feature_one_sixteen, fpn_output_channels * height_16 * width_16, cout_shifts[54-1]);
    // save_layer<qaint>(save_dir, "feature_one_eight", filename, reference_feature_one_eight, fpn_output_channels * height_8 * width_8, cout_shifts[56-1]);
    // save_layer<qaint>(save_dir, "feature_half", filename, reference_feature_half, fpn_output_channels * height_2 * width_2, cout_shifts[conv_cnt-1]);

    // if (n_measurement_frames == 0) return;

    // qaint cost_volume[n_depth_levels * height_2 * width_2];
    // cost_volume_fusion(reference_feature_half, n_measurement_frames, measurement_feature_halfs, warpings, cost_volume);
    // save_layer<qaint>(save_dir, "cost_volume", filename, cost_volume, n_depth_levels * height_2 * width_2, cin_shifts[conv_cnt]);

    // qaint skip0[hyper_channels * height_2 * width_2];
    // qaint skip1[(hyper_channels * 2) * height_4 * width_4];
    // qaint skip2[(hyper_channels * 4) * height_8 * width_8];
    // qaint skip3[(hyper_channels * 8) * height_16 * width_16];
    // qaint bottom[(hyper_channels * 16) * height_32 * width_32];
    // CostVolumeEncoder(reference_feature_half, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen, cost_volume,
    //                   skip0, skip1, skip2, skip3, bottom, filename);
    // save_layer<qaint>(save_dir, "skip0", filename, skip0, hyper_channels * height_2 * width_2, oout_shifts[39-1]);
    // save_layer<qaint>(save_dir, "skip1", filename, skip1, (hyper_channels * 2) * height_4 * width_4, oout_shifts[43-1]);
    // save_layer<qaint>(save_dir, "skip2", filename, skip2, (hyper_channels * 4) * height_8 * width_8, oout_shifts[47-1]);
    // save_layer<qaint>(save_dir, "skip3", filename, skip3, (hyper_channels * 8) * height_16 * width_16, oout_shifts[51-1]);
    // save_layer<qaint>(save_dir, "bottom", filename, bottom, (hyper_channels * 16) * height_32 * width_32, oout_shifts[other_cnt-1]);

    // save_layer<qaint>(save_dir, "cell_state_prev", filename, cell_state, hid_channels * height_32 * width_32, cellshift);
    // save_layer<qaint>(save_dir, "hidden_state_prev", filename, hidden_state, hid_channels * height_32 * width_32, oin_shifts[other_cnt]);
    // LSTMFusion(bottom, hidden_state, cell_state, filename);
    // save_layer<qaint>(save_dir, "cell_state", filename, cell_state, hid_channels * height_32 * width_32, cellshift);
    // save_layer<qaint>(save_dir, "hidden_state", filename, hidden_state, hid_channels * height_32 * width_32, oin_shifts[other_cnt]);

    // CostVolumeDecoder(reference_image, skip0, skip1, skip2, skip3, hidden_state, depth_full);
    // save_layer<qaint>(save_dir, "depth_full", filename, depth_full, test_image_height * test_image_width, sigshift);
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
    qaint hidden_state[hid_channels * height_32 * width_32];
    qaint cell_state[hid_channels * height_32 * width_32];

    ofstream ofs;

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

        float reference_image_float[3 * test_image_height * test_image_width];
        load_image(image_filenames[f], reference_image_float);
        qaint reference_image[3 * test_image_height * test_image_width];
        const int ashift = cin_shifts[0];
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
                in_hidden_state[i][j][k] = hidden_state[(i * height_32 + j) * width_32 + k] / (float) (1 << hiddenshift);
            float out_hidden_state[hid_channels][height_32][width_32];
            warp_from_depth(in_hidden_state, depth_estimation[0], trans, lstm_K_bottom, out_hidden_state);

            for (int i = 0; i < hid_channels; i++) for (int j = 0; j < height_32; j++) for (int k = 0; k < width_32; k++)
                hidden_state[(i * height_32 + j) * width_32 + k] = (depth_estimation[0][j][k] <= 0.01) ? 0.0 : out_hidden_state[i][j][k] * (1 << hiddenshift);
        }

        qaint reference_feature_half[fpn_output_channels * height_2 * width_2];
        qaint depth_full[test_image_height * test_image_width];
        predict(reference_image, n_measurement_frames, measurement_feature_halfs,
                warpings, reference_feature_half, hidden_state, cell_state, depth_full, image_filenames[f].substr(len_image_filedir, 5));
        delete[] warpings;
        break;

        keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half);
        if (response == 0) continue;

        float prediction[test_image_height * test_image_width];
        for (int idx = 0; idx < test_image_height * test_image_width; idx++)
            prediction[idx] = 1.0 / (inverse_depth_multiplier * (depth_full[idx] / (float) (1 << sigshift)) + inverse_depth_base);

        for (int i = 0 ; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++)
            previous_depth[i][j] = prediction[i * test_image_width + j];
        for (int i = 0 ; i < 4; i++) for (int j = 0; j < 4; j++)
            previous_pose[i * 4 + j] = reference_pose[i * 4 + j];
        previous_exists = true;

        state_exists = true;

        save_image(save_dir + image_filenames[f].substr(len_image_filedir), previous_depth);

        ofs.open(save_dir + image_filenames[f].substr(len_image_filedir, 5) + ".txt");
        for (int i = 0 ; i < test_image_height; i++) {
            for (int j = 0; j < test_image_width-1; j++)
                ofs << previous_depth[i][j] << " ";
            ofs << previous_depth[i][test_image_width-1] << "\n";
        }
        ofs.close();
    }

    keyframe_buffer.close();

    delete[] weights;
    delete[] biases;
    delete[] scales;

    return 0;
}
