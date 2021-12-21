#pragma once
#include "settings.h"

// const int org_image_width = 540;
// const int org_image_height = 360;
const int org_image_width = 30;
const int org_image_height = 40;

// const int test_image_width = 320;
// const int test_image_height = 256;
const int test_image_width = 10;
const int test_image_height = 20;
const int test_distortion_crop = 0;
const bool test_perform_crop = false;
const bool test_visualize = true;
const int test_n_measurement_frames = 2;
const int test_keyframe_buffer_size = 30;
const float test_keyframe_pose_distance = 0.1;
const float test_optimal_t_measure = 0.15;
const float test_optimal_R_measure = 0.0;

const int warp_grid_width = test_image_width / 2;
const int warp_grid_height = test_image_height / 2;
// const int warp_grid_size = warp_grid_width * warp_grid_height;

// SET THESE: TRAINING FOLDER LOCATIONS
const string dataset = "/home/nhsmt1123/master-thesis/deep-video-mvs/data/7scenes";
const string train_run_directory = "/home/nhsmt1123/master-thesis/deep-video-mvs/training-runs";

const string fusionnet_train_weights = "weights";
const string fusionnet_test_weights = "weights";

// SET THESE: TESTING FOLDER LOCATIONS
// for run-testing-online.py (evaluate a single scene, WITHOUT keyframe indices, online selection)
const string test_online_scene_path = "/home/nhsmt1123/master-thesis/deep-video-mvs/sample-data/hololens-dataset/000";

const string dataset_name = "hololens-dataset";
const string system_name = "keyframe_" + dataset_name + "_" + to_string(test_image_width) + "_" + to_string(test_image_height) + "_" + to_string(test_n_measurement_frames) + "_dvmvs_fusionnet_online";

const string scene_folder = test_online_scene_path;
const string scene = "000";

const int n_test_frames = 100;

const float scale_rgb = 255.0;
const float mean_rgb[3] = {0.485, 0.456, 0.406};
const float std_rgb[3] = {0.229, 0.224, 0.225};

const float min_depth = 0.25;
const float max_depth = 20.0;
const int n_depth_levels = 64;

const int fpn_output_channels = 32;
const int hyper_channels = 32;

#define conv_out_size(size, kernel_size, stride, padding) ((size) + 2 * (padding) - (kernel_size)) / (stride) + 1

#define fe1_out_size(size) conv_out_size(conv_out_size(conv_out_size((size), 3, 2, 1), 3, 1, 1), 1, 1, 0)
#define fe2_out_size(size) stack_out_size(fe1_out_size((size)), 3, 2)
#define fe3_out_size(size) stack_out_size(fe2_out_size((size)), 5, 2)
#define fe4_out_size(size) stack_out_size(stack_out_size(fe3_out_size((size)), 5, 2), 3, 1)
#define fe5_out_size(size) stack_out_size(stack_out_size(fe4_out_size((size)), 5, 2), 3, 1)

#define fe1_out_channels 16
#define fe2_out_channels 24
#define fe3_out_channels 40
#define fe4_out_channels 96
#define fe5_out_channels 320

// #define conv_layer_out_size(size, kernel_size, stride) conv_out_size((size), (kernel_size), (stride), ((kernel_size) - 1) / 2)
// #define down_conv_out_size(size) conv_layer_out_size((size), (kernel_size), 2)
// #define eb_out_size(size, kernel_size) down_conv_out_size((size), (kernel_size))

// utils
void pose_distance(const float reference_pose[4][4], const float measurement_pose[4][4], float &combined_measure, float &R_measure, float &t_measure);
void get_warp_grid_for_cost_volume_calculation(float warp_grid[3][warp_grid_width * warp_grid_height]);
void cost_volume_fusion(const float image1[fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)],
                        const float image2s[test_n_measurement_frames][fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)],
                        const float pose1[4][4],
                        const float pose2s[test_n_measurement_frames][4][4],
                        const float K[3][3],
                        const float warp_grid[3][warp_grid_width * warp_grid_height],
                        const int n_measurement_frames,
                        float fused_cost_volume[n_depth_levels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)]);
bool is_pose_available(const float pose[4][4]);

// keyframe_buffer
class KeyframeBuffer{
public:
    int try_new_keyframe(const float pose[4][4], const float image[org_image_height][org_image_width][3]);
    int get_best_measurement_frames(float measurement_poses[test_n_measurement_frames][4][4],
                                    float measurement_images[test_n_measurement_frames][org_image_height][org_image_width][3]);

private:
    const int buffer_size = test_keyframe_buffer_size;
    int buffer_idx = 0;
    int buffer_cnt = 0;
    float buffer_poses[test_keyframe_buffer_size][4][4];
    float buffer_images[test_keyframe_buffer_size][org_image_height][org_image_width][3];
    const float optimal_R_score = test_optimal_R_measure;
    const float optimal_t_score = test_optimal_t_measure;
    const float keyframe_pose_distance = test_keyframe_pose_distance;
    int __tracking_lost_counter = 0;
    float calculate_penalty(const float t_score, const float R_score);
};

// dataset_loader
void load_image(string image_filename, float reference_image[org_image_height][org_image_width][3]);

class PreprocessImage{
public:
    PreprocessImage(float K[3][3]) {
        float factor_x = (float) test_image_width / (float) org_image_width;
        float factor_y = (float) test_image_height / (float) org_image_height;
        fx = K[0][0] * factor_x;
        fy = K[1][1] * factor_y;
        cx = K[0][2] * factor_x;
        cy = K[1][2] * factor_y;
    }
    void apply_rgb(float image[org_image_height][org_image_width][3], float resized_image[3][test_image_height][test_image_width]);
    void get_updated_intrinsics(float updated_intrinsic[3][3]);

private:
    float fx, fy, cx, cy;
};
