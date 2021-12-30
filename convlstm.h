#pragma once
#include "config.h"
#include "conv.h"
// #include "batchnorm.h"
// #include "activation.h"

template <int in_channels, int hid_channels, int height, int width, int kernel_size>
class MVSLayernormConvLSTMCell{
public:
    MVSLayernormConvLSTMCell(const string param_path) : param_path(param_path) {}

    void forward(const float input[in_channels][height][width],
                 const bool previous_exists,
                 const float previous_pose[4][4],
                 const float current_pose[4][4],
                 const float estimated_current_depth[height][width],
                 const float camera_matrix[3][3],
                 const bool state_exists,
                 float hidden_state[hyper_channels * 16][height][width],
                 float cell_state[hyper_channels * 16][height][width]) {

        const int l0_in_channels = in_channels + hid_channels;
        const int l0_out_channels = 4 * hid_channels;

        if (!previous_exists) {
            Matrix4f p_pose, c_pose;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) p_pose(i, j) = previous_pose[i][j];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c_pose(i, j) = current_pose[i][j];

            Matrix4f transformation = p_pose.inverse() * c_pose;

            float tmp_hidden_state[hid_channels][height][width];
            warp_from_depth(hidden_state, estimated_current_depth, transformation, camera_matrix, tmp_hidden_state);

            for (int i = 0; i < hid_channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
                hidden_state[i][j][k] = (estimated_current_depth[j][k] <= 0.01) ? 0.0 : tmp_hidden_state[i][j][k];
        }

        float combined[l0_in_channels][height][width];
        for (int i = 0; i < in_channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            combined[i][j][k] = input[i][j][k];
        for (int i = 0; i < hid_channels i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            combined[i + in_channels][j][k] = hidden_state[i][j][k];

        const int stride = 1;
        const int padding = (kernel_size - 1) / 2;
        const int groups = 1;
        Conv2d<l0_in_channels, height, width, l0_out_channels, height, width, kernel_size, stride, padding, groups> l0_conv(param_path + ".conv");

        float combined_conv[l0_in_channels][height][width];
        l0_conv.forward(combined, combined_conv);

        float ii[hid_channels][height][width];
        float ff[hid_channels][height][width];
        float oo[hid_channels][height][width];
        float gg[hid_channels][height][width];
        for (int i = 0; i < hid_channels i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            ii[i][j][k] = combined_conv[i][j][k];
        for (int i = 0; i < hid_channels i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            ff[i][j][k] = combined_conv[i+hid_channels][j][k];
        for (int i = 0; i < hid_channels i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            oo[i][j][k] = combined_conv[i+hid_channels*2][j][k];
        for (int i = 0; i < hid_channels i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            gg[i][j][k] = combined_conv[i+hid_channels*3][j][k];

        Sigmoid<hid_channels, height, width> l1_sigmoid;
        celu<hid_channels, height, width> l1_celu;

        l1_sigmoid.forward(ii, ii);
        l1_sigmoid.forward(ff, ff);
        l1_sigmoid.forward(oo, oo);

        layer_norm<hid_channels, height, width>(gg, gg);
        l1_celu.forward(gg, gg);

        for (int i = 0; i < hid_channels i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            cell_state[i][j][k] = ff[i][j][k] * cell_state[i][j][k] + ii[i][j][k] * gg[i][j][k];

        layer_norm<hid_channels, height, width>(cell_state, cell_state);
        l1_celu.forward(cell_state, hidden_state);

        for (int i = 0; i < hid_channels i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            hidden_state[i][j][k] *= oo[i][j][k];

    }

private:
    string param_path;
};

