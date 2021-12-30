#pragma once
#include "config.h"
#include "conv.h"
// #include "batchnorm.h"
// #include "activation.h"

template <int in_height, int in_width, int kernel_size>
class MVSLayernormConvLSTMCell{
public:
    MVSLayernormConvLSTMCell(const string param_path) : param_path(param_path) {}

    void forward(const float input[hyper_channels * 16][fe5_out_size(test_image_height)][fe5_out_size(test_image_width)],
                 const bool state_exists,
                 const float h_cur[hyper_channels * 16][fe5_out_size(test_image_height)][fe5_out_size(test_image_width)],
                 const float c_cur[hyper_channels * 2][fe2_out_size(in_height)][fe2_out_size(in_width)],
                 const bool previous_exists,
                 const float previous_pose[4][4],
                 const float current_pose[4][4],
                 const float skip2[hyper_channels * 4][fe3_out_size(in_height)][fe3_out_size(in_width)],
                 const float skip3[hyper_channels * 8][fe4_out_size(in_height)][fe4_out_size(in_width)],
                 const float bottom[hyper_channels * 16][fe5_out_size(in_height)][fe5_out_size(in_width)],
                 float depth_full[in_height][in_width]) {

        float combined[hyper_channels * 32][fe5_out_size(test_image_height)][fe5_out_size(test_image_width)];
        if (!previous_exists) {
            Matrix4f p_pose, c_pose;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) p_pose(i, j) = previous_pose[i][j];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c_pose(i, j) = current_pose[i][j];

            Matrix4f transformation = p_pose.inverse() * c_pose;

        } else {
            if  (int i = 0; i < hyper_channels * 16; i++) for (int j = 0; j < fe5_out_size(test_image_height); j++) for (int k = 0; k < fe5_out_size(test_image_width); k++)
                combined[i][j][k] = input[i][j][k];
            if  (int i = 0; i < hyper_channels * 16; i++) for (int j = 0; j < fe5_out_size(test_image_height); j++) for (int k = 0; k < fe5_out_size(test_image_width); k++)
                combined[i + hyper_channels * 16][j][k] = h_cur[i][j][k];
        }

        const int stride = 1;
        const int padding = (kernel_size - 1) / 2;
        const int groups = 1;

        const int l0_in_channels = in_channels + hid_channels;
        const int l0_out_channels = 4 * hid_channels;
        Conv2d<l0_in_channels, fe1_out_size(in_height), fe1_out_size(in_width), l0_out_channels, fe1_out_size(in_height), fe1_out_size(in_width), kernel_size, stride, padding, groups> l0_conv(param_path + ".conv");

    }

private:
    string param_path;
};

