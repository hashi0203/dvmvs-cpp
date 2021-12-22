#pragma once
#include "config.h"
#include "conv.h"
#include "batchnorm.h"
#include "activation.h"

#define invres_out_size(size, kernel_size, stride) conv_out_size((size), (kernel_size), (stride), (kernel_size) / 2)
#define stack_out_size(size, kernel_size, stride) invres_out_size((size), (kernel_size), (stride))

template <int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int kernel_size, int stride, int expansion_factor>
class _InvertedResidual{
public:

    _InvertedResidual(const string param_path) : param_path(param_path) {}

    void forward(const float x[in_channels][in_height][in_width], float y[out_channels][out_height][out_width]) {
        const int mid_channels = in_channels * expansion_factor;

        // Pointwise
        const int l0_kernel_size = 1;
        const int l0_stride = 1;
        const int l0_padding = 0;
        const int l0_groups = 1;
        const int l0_out_channels = mid_channels;
        const int l0_out_height = conv_out_size(in_height, l0_kernel_size, l0_stride, l0_padding);
        const int l0_out_width = conv_out_size(in_width, l0_kernel_size, l0_stride, l0_padding);
        Conv2d<in_channels, in_height, in_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size, l0_stride, l0_padding, l0_groups> l0_conv(param_path + ".layers.0");

        const int l1_out_channels = mid_channels;
        const int l1_out_height = l0_out_height;
        const int l1_out_width = l0_out_width;
        BatchNorm2d<l1_out_channels, l1_out_height, l1_out_width> l1_bn(param_path + ".layers.1");

        const int l2_out_channels = mid_channels;
        const int l2_out_height = l1_out_height;
        const int l2_out_width = l1_out_width;
        ReLU<l2_out_channels, l2_out_height, l2_out_width> l2_relu;

        // Depthwise
        const int l3_kernel_size = kernel_size;
        const int l3_stride = stride;
        const int l3_padding = kernel_size / 2;
        const int l3_groups = mid_channels;
        const int l3_out_channels = mid_channels;
        const int l3_out_height = conv_out_size(l2_out_height, l3_kernel_size, l3_stride, l3_padding);
        const int l3_out_width = conv_out_size(l2_out_width, l3_kernel_size, l3_stride, l3_padding);
        Conv2d<l2_out_channels, l2_out_height, l2_out_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size, l3_stride, l3_padding, l3_groups> l3_conv(param_path + ".layers.3");

        const int l4_out_channels = mid_channels;
        const int l4_out_height = l3_out_height;
        const int l4_out_width = l3_out_width;
        BatchNorm2d<l4_out_channels, l4_out_height, l4_out_width> l4_bn(param_path + ".layers.4");

        const int l5_out_channels = mid_channels;
        const int l5_out_height = l4_out_height;
        const int l5_out_width = l4_out_width;
        ReLU<l5_out_channels, l5_out_height, l5_out_width> l5_relu;

        // Linear pointwise. Note that there's no activation.
        const int l6_kernel_size = 1;
        const int l6_stride = 1;
        const int l6_padding = 0;
        const int l6_groups = 1;
        const int l6_out_channels = out_channels;
        const int l6_out_height = conv_out_size(l5_out_height, l6_kernel_size, l6_stride, l6_padding);
        const int l6_out_width = conv_out_size(l5_out_width, l6_kernel_size, l6_stride, l6_padding);
        Conv2d<l5_out_channels, l5_out_height, l5_out_width, l6_out_channels, l6_out_height, l6_out_width, l6_kernel_size, l6_stride, l6_padding, l6_groups> l6_conv(param_path + ".layers.6");

        const int l7_out_channels = out_channels;
        const int l7_out_height = l6_out_height;
        const int l7_out_width = l6_out_width;
        BatchNorm2d<l7_out_channels, l7_out_height, l7_out_width> l7_bn(param_path + ".layers.7");

        float y0[l0_out_channels][l0_out_height][l0_out_width];
        // float ***y0 = new float**[l0_out_channels];
        // new_3d(y0, l0_out_channels, l0_out_height, l0_out_width);
        l0_conv.forward(x, y0);
        // l0_conv.close();

        // float y1[l1_out_channels][l1_out_height][l1_out_width];
        l1_bn.forward(y0, y0);
        // l1_bn.close();

        // float y2[l2_out_channels][l2_out_height][l2_out_width];
        l2_relu.forward(y0, y0);

        float y3[l3_out_channels][l3_out_height][l3_out_width];
        // float ***y3 = new float**[l3_out_channels];
        // new_3d(y3, l3_out_channels, l3_out_height, l3_out_width);
        l3_conv.forward(y0, y3);
        // delete_3d(y0, l0_out_channels, l0_out_height, l0_out_width);
        // l3_conv.close();

        // float y4[l4_out_channels][l4_out_height][l4_out_width];
        l4_bn.forward(y3, y3);
        // l4_bn.close();

        // float y5[l5_out_channels][l5_out_height][l5_out_width];
        l5_relu.forward(y3, y3);

        float y6[l6_out_channels][l6_out_height][l6_out_width];
        // float ***y6 = new float**[l6_out_channels];
        // new_3d(y6, l6_out_channels, l6_out_height, l6_out_width);
        l6_conv.forward(y3, y6);
        // delete_3d(y3, l3_out_channels, l3_out_height, l3_out_width);
        // l6_conv.close();

        // float y7[l7_out_channels][l7_out_height][l7_out_width];
        l7_bn.forward(y6, y);
        // delete_3d(y6, l6_out_channels, l6_out_height, l6_out_width);
        // l7_bn.close();

        // if x.shape == y.shape
        if (in_channels == out_channels && stride == 1) {
            for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
                y[i][j][k] += x[i][j][k];
        }
        // if (in_channels == out_channels && stride == 1) {
        //     for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
        //         y[i][j][k] = y7[i][j][k] + x[i][j][k];
        // } else {
        //     for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
        //         y[i][j][k] = y7[i][j][k];
        // }
    }

private:
    string param_path;
};


template <int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int kernel_size, int stride, int expansion_factor, int repeats>
class _stack{
// Creates a stack of inverted residuals.
public:
    _stack(const string param_path) : param_path(param_path) {}

    void forward(float x[in_channels][in_height][in_width], float y[out_channels][out_height][out_width]) {
        // First one has no skip, because feature map size changes.
        _InvertedResidual<in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, expansion_factor> l0_invres(param_path + ".0");

        float yi[out_channels][out_height][out_width];
        l0_invres.forward(x, yi);

        for (int i = 1; i < repeats; i++) {
            _InvertedResidual<out_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, 1, expansion_factor> li_invres(param_path + "." + to_string(i));
            li_invres.forward(yi, y);
            for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
                yi[i][j][k] = y[i][j][k];
        }
    }

private:
    string param_path;
};
