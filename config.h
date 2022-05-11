#pragma once
#include "settings.h"

#include <unordered_map>

constexpr int qwbit = 8;
typedef short qwint;
constexpr int qabit = 16;
typedef int qaint;
// typedef short qaint;
// constexpr qaint QA_MIN = -32768;

// constexpr int bufbit = 8;
typedef long long qmint;

constexpr int org_image_width = 540;
constexpr int org_image_height = 360;
// constexpr int test_image_width = 320;
// constexpr int test_image_height = 256;

// constexpr int org_image_width = 128;
// constexpr int org_image_height = 128;
constexpr int test_image_width = 96;
constexpr int test_image_height = 64;

constexpr int test_n_measurement_frames = 2;
constexpr int test_keyframe_buffer_size = 30;
constexpr float test_keyframe_pose_distance = 0.1;
constexpr float test_optimal_t_measure = 0.15;
constexpr float test_optimal_R_measure = 0.0;

// SET THESE: TESTING FOLDER LOCATIONS
// for run-testing-online.py (evaluate a single scene, WITHOUT keyframe indices, online selection)
const string test_online_scene_path = "/home/nhsmt1123/master-thesis/deep-video-mvs/sample-data/hololens-dataset/000";

const string dataset_name = "hololens-dataset";
const string system_name = "keyframe_" + dataset_name + "_" + to_string(test_image_width) + "_" + to_string(test_image_height) + "_" + to_string(test_n_measurement_frames) + "_dvmvs_fusionnet_online";

const string scene_folder = test_online_scene_path;
const string scene = "000";

constexpr int n_test_frames = 20;

constexpr float scale_rgb = 255.0;
constexpr float mean_rgb[3] = {0.485, 0.456, 0.406};
constexpr float std_rgb[3] = {0.229, 0.224, 0.225};

constexpr float min_depth = 0.25;
constexpr float max_depth = 20.0;
constexpr int n_depth_levels = 64;

constexpr float inverse_depth_base = 1.0 / max_depth;
constexpr float inverse_depth_multiplier = 1.0 / min_depth - 1.0 / max_depth;
constexpr float inverse_depth_step = inverse_depth_multiplier / (n_depth_levels - 1);

constexpr int fpn_output_channels = 32;
constexpr int hyper_channels = 32;
constexpr int hid_channels = hyper_channels * 16;

constexpr int channels_1 = 16;
constexpr int channels_2 = 24;
constexpr int channels_3 = 40;
constexpr int channels_4 = 96;
constexpr int channels_5 = 320;

constexpr int height_2 = test_image_height / 2;
constexpr int height_4 = test_image_height / 4;
constexpr int height_8 = test_image_height / 8;
constexpr int height_16 = test_image_height / 16;
constexpr int height_32 = test_image_height / 32;

constexpr int width_2 = test_image_width / 2;
constexpr int width_4 = test_image_width / 4;
constexpr int width_8 = test_image_width / 8;
constexpr int width_16 = test_image_width / 16;
constexpr int width_32 = test_image_width / 32;

constexpr float height_normalizer = height_2 / 2.0;
constexpr float width_normalizer = width_2 / 2.0;

extern qwint* params;
constexpr int n_files = 255 + 18 + 80 + 1 + 80;
extern int start_idx[n_files + 1];
extern int param_cnt;
extern int shifts[n_files];
constexpr int offsets[81] = {20249, 20272, 25322, 27197, 26266, 28856, 20246, 20327, 45463, 25185, 20337, 29676, 28738, 18059, 35674, 32424, 20304, 28114, 20252, 20316, 36747, 23606, 34667, 32653, 20258, 20359, 35470, 20253, 20328, 44308, 20206, 20324, 32374, 20261, 20482, 40524, 20259, 20300, 32128, 20261, 20390, 38267, 20258, 20342, 43511, 20254, 20353, 35854, 20252, 20371, 49239, 28554, 21445, 27444, 27512, 32646, 39416, 20982, 24687, 30980, 24178, 28403, 28962, 35834, 22230, 33524, 39682, 32411, 20297, 29811, 23930, 25626, 27700, 20613, 19562, 34279, 26256, 19968, 35764, 23113, 24952};
// constexpr int offsets[81] = {10125, 10137, 12661, 13599, 13133, 14428, 10123, 10164, 22731, 12593, 10169, 14838, 14369, 9029, 17837, 16212, 10152, 14057, 10126, 10158, 18373, 11803, 17334, 16327, 10129, 10180, 17735, 10127, 10164, 22154, 10103, 10162, 16187, 10130, 10242, 20262, 10130, 10151, 16064, 10130, 10196, 19134, 10129, 10171, 21756, 10127, 10177, 17927, 10126, 10186, 24620, 14277, 10723, 13722, 13756, 16323, 19708, 10492, 12344, 15490, 12089, 14202, 14481, 17917, 11115, 16763, 19841, 16206, 10149, 14905, 11965, 12813, 13850, 10307, 9781, 17140, 13128, 9984, 17882, 11557, 12477};
// constexpr int offsets[81] = {5062, 5068, 6331, 6799, 6567, 7214, 5062, 5082, 11366, 6297, 5084, 7419, 7185, 4515, 8919, 8106, 5076, 7029, 5063, 5079, 9187, 5901, 8667, 8164, 5064, 5090, 8868, 5063, 5082, 11077, 5052, 5081, 8094, 5065, 5121, 10131, 5065, 5075, 8032, 5065, 5098, 9567, 5064, 5086, 10878, 5064, 5088, 8964, 5063, 5093, 12310, 7138, 5362, 6861, 6878, 8162, 9854, 5246, 6172, 7745, 6045, 7101, 7241, 8958, 5558, 8381, 9921, 8103, 5074, 7453, 5983, 6407, 6925, 5154, 4890, 8570, 6564, 4992, 8941, 5778, 6238};
// constexpr int offsets[81] = {1266, 1267, 1583, 1700, 1642, 1804, 1266, 1271, 2842, 1574, 1271, 1855, 1797, 1129, 2230, 2027, 1269, 1757, 1266, 1270, 2297, 1475, 2167, 2041, 1266, 1273, 2217, 1266, 1271, 2770, 1263, 1271, 2024, 1267, 1280, 2533, 1267, 1269, 2008, 1267, 1275, 2392, 1266, 1272, 2720, 1266, 1272, 2241, 1266, 1274, 3078, 1785, 1341, 1716, 1720, 2041, 2464, 1312, 1543, 1937, 1512, 1776, 1810, 2240, 1390, 2096, 2480, 2026, 1269, 1863, 1496, 1602, 1731, 1289, 1223, 2143, 1641, 1248, 2236, 1445, 1560};
// constexpr int offsets[81] = {2531, 2534, 3165, 3400, 3284, 3607, 2531, 2541, 5683, 3148, 2542, 3710, 3592, 2258, 4460, 4053, 2538, 3514, 2532, 2540, 4593, 2951, 4334, 4082, 2533, 2545, 4434, 2532, 2541, 5539, 2526, 2541, 4047, 2533, 2561, 5066, 2533, 2538, 4016, 2533, 2549, 4784, 2533, 2543, 5439, 2532, 2545, 4482, 2532, 2547, 6155, 3570, 2681, 3431, 3439, 4081, 4928, 2623, 3086, 3873, 3022, 3551, 3621, 4480, 2779, 4191, 4961, 4052, 2537, 3727, 2992, 3204, 3463, 2577, 2446, 4285, 3282, 2496, 4471, 2889, 3119};
// constexpr int offsets[81] = {316, 317, 396, 425, 411, 451, 316, 318, 710, 394, 318, 464, 449, 283, 558, 507, 317, 440, 316, 317, 575, 369, 542, 511, 317, 318, 554, 316, 318, 692, 316, 318, 506, 317, 320, 633, 317, 317, 502, 317, 319, 598, 317, 318, 680, 316, 318, 561, 316, 318, 770, 446, 335, 429, 430, 510, 616, 328, 386, 484, 378, 444, 453, 560, 348, 524, 620, 507, 318, 466, 374, 401, 433, 322, 306, 536, 411, 312, 559, 362, 390};
extern int offset_cnt;
extern float* params_f;
const int n_acts = 178;
extern int actshifts[n_acts];
extern int act_cnt;


#define conv_out_size(size, kernel_size, stride, padding) ((size) + 2 * (padding) - (kernel_size)) / (stride) + 1
#define invres_out_size(size, kernel_size, stride) conv_out_size((size), (kernel_size), (stride), (kernel_size) / 2)
#define stack_out_size(size, kernel_size, stride) invres_out_size((size), (kernel_size), (stride))

#define new_2d_qaint(arr, d0, d1) for (int iii2 = 0; iii2 < (d0); iii2++) {(arr)[iii2] = new qaint[(d1)];}
#define new_2d(arr, d0, d1) for (int iii2 = 0; iii2 < (d0); iii2++) {(arr)[iii2] = new float[(d1)];}
#define new_3d(arr, d0, d1, d2) for (int iii3 = 0; iii3 < (d0); iii3++) {(arr)[iii3] = new float*[(d1)]; new_2d((arr)[iii3], (d1), (d2));}
#define new_4d(arr, d0, d1, d2, d3) for (int iii4 = 0; iii4 < (d0); iii4++) {(arr)[iii4] = new float**[(d1)]; new_3d((arr)[iii4], (d1), (d2), (d3));}

#define tmp_delete_2d(arr, d0, d1) for (int iii2 = 0; iii2 < (d0); iii2++) {delete[] (arr)[iii2];}
#define tmp_delete_3d(arr, d0, d1, d2) for (int iii3 = 0; iii3 < (d0); iii3++) {tmp_delete_2d((arr)[iii3], (d1), (d2)); delete[] (arr)[iii3];}
#define tmp_delete_4d(arr, d0, d1, d2, d3) for (int iii4 = 0; iii4 < (d0); iii4++) {tmp_delete_3d((arr)[iii4], (d1), (d2), (d3)); delete[] (arr)[iii4];}

#define delete_2d(arr, d0, d1) tmp_delete_2d(arr, d0, d1) ; delete[] (arr);
#define delete_3d(arr, d0, d1, d2) tmp_delete_3d(arr, d0, d1, d2) ; delete[] (arr);
#define delete_4d(arr, d0, d1, d2, d3) tmp_delete_4d(arr, d0, d1, d2, d3); delete[] (arr);


// utils
void pose_distance(const float reference_pose[4 * 4], const float measurement_pose[4 * 4], float &combined_measure, float &R_measure, float &t_measure);
void get_warp_grid_for_cost_volume_calculation(float warp_grid[3][width_2 * height_2]);
void cost_volume_fusion(const float image1[fpn_output_channels * height_2 * width_2],
                        const int n_measurement_frames,
                        const float image2s[test_n_measurement_frames * fpn_output_channels * height_2 * width_2],
                        const float* warpings,
                        float fused_cost_volume[n_depth_levels * height_2 * width_2]);
void get_non_differentiable_rectangle_depth_estimation(const float reference_pose[4 * 4],
                                                       const float measurement_pose[4 * 4],
                                                       const float previous_depth[test_image_height][test_image_width],
                                                       const float full_K[3][3],
                                                       const float half_K[3][3],
                                                       float depth_hypothesis[1][height_2][width_2]);
void warp_from_depth(const float image_src[hyper_channels * 16][height_32][width_32],
                     const float depth_dst[height_32][width_32],
                     const float trans[4][4],
                     const float camera_matrix[3][3],
                     float image_dst[hyper_channels * 16][height_32][width_32]);
bool is_pose_available(const float pose[4 * 4]);

// keyframe_buffer
class KeyframeBuffer{
public:
    KeyframeBuffer(){
        new_2d(buffer_poses, buffer_size, 4 * 4);
        new_2d_qaint(buffer_feature_halfs, buffer_size, fpn_output_channels * height_2 * width_2);
    }

    int try_new_keyframe(const float pose[4 * 4]);
    void add_new_keyframe(const float pose[4 * 4], const qaint feature_half[fpn_output_channels * height_2 * width_2]);
    int get_best_measurement_frames(const float reference_pose[4 * 4], float measurement_poses[test_n_measurement_frames * 4 * 4], qaint measurement_feature_halfs[test_keyframe_buffer_size * fpn_output_channels * height_2 * width_2]);

    void close() {
        delete_2d(buffer_poses, buffer_size, 4 * 4);
        delete_2d(buffer_feature_halfs, buffer_size, fpn_output_channels * height_2 * width_2);
    }

private:
    const int buffer_size = test_keyframe_buffer_size;
    int buffer_idx = 0;
    int buffer_cnt = 0;
    float **buffer_poses = new float*[test_keyframe_buffer_size];
    qaint **buffer_feature_halfs = new qaint*[test_keyframe_buffer_size];
    const float optimal_R_score = test_optimal_R_measure;
    const float optimal_t_score = test_optimal_t_measure;
    const float keyframe_pose_distance = test_keyframe_pose_distance;
    int __tracking_lost_counter = 0;
    float calculate_penalty(const float t_score, const float R_score);
};

// dataset_loader
void get_updated_intrinsics(const float K[3][3], float updated_intrinsic[3][3]);
void load_image(const string image_filename, float reference_image[3 * test_image_height * test_image_width]);
void save_image(const string image_filename, float depth[test_image_height][test_image_width]);
