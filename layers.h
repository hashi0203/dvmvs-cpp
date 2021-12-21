#pragma once
#include "config.h"
#include "conv.h"
#include "batchnorm.h"
#include "activation.h"

template <int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int kernel_size, int stride, bool apply_bn_relu>
class conv_layer{
public:
    conv_layer(const string param_path) : param_path(param_path) {}

    void forward(const float x[in_channels][in_height][in_width], float y[out_channels][out_height][out_width]) {
        const int padding = (kernel_size - 1) / 2;
        const int groups = 1;
        Conv2d<in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, padding, groups> l0_conv(param_path + ".0");

        // float y0[out_channels][out_height][out_width];
        l0_conv.forward(x, y);

        if (apply_bn_relu) {
            BatchNorm2d<out_channels, out_height, out_width> l1_bn(param_path + ".1");
            // float y1[out_channels][out_height][out_width];
            l1_bn.forward(y, y);

            ReLU<out_channels, out_height, out_width> l2_relu;
            // float y2[out_channels][out_height][out_width];
            l2_relu.forward(y, y);

        //     for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
        //         y[i][j][k] = y2[i][j][k];
        // } else {
        //     for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
        //         y[i][j][k] = y0[i][j][k];
        }
    }

private:
    string param_path;
};
