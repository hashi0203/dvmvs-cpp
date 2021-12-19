#pragma once
#include "config.h"
#include "torch.h"
#include "conv.h"
#include "batchnorm.h"
#include "activation.h"

#define invres_out_size(size, kernel_size, stride) conv_out_size((size), (kernel_size), (stride), (kernel_size) / 2)
#define stack_out_size(size, kernel_size, stride) invres_out_size((size), (kernel_size), (stride))

template <int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int kernel_size, int stride, int expansion_factor>
class _InvertedResidual{
public:

    _InvertedResidual(const string param_path) : param_path(param_path) {}

    void forward(float x[in_channels][in_height][in_width], float y[out_channels][out_height][out_width]) {
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
        l0_conv.forward(x, y0);

        float y1[l1_out_channels][l1_out_height][l1_out_width];
        l1_bn.forward(y0, y1);

        float y2[l2_out_channels][l2_out_height][l2_out_width];
        l2_relu.forward(y1, y2);

        float y3[l3_out_channels][l3_out_height][l3_out_width];
        l3_conv.forward(y2, y3);

        float y4[l4_out_channels][l4_out_height][l4_out_width];
        l4_bn.forward(y3, y4);

        float y5[l5_out_channels][l5_out_height][l5_out_width];
        l5_relu.forward(y4, y5);

        float y6[l6_out_channels][l6_out_height][l6_out_width];
        l6_conv.forward(y5, y6);

        float y7[l7_out_channels][l7_out_height][l7_out_width];
        l7_bn.forward(y6, y7);

        // if x.shape == y.shape
        if (in_channels == out_channels && stride == 1) {
            for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
                y[i][j][k] = y7[i][j][k] + x[i][j][k];
        } else {
            for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
                y[i][j][k] = y7[i][j][k];
        }
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


// template <int in_channels, int in_height, int in_width>
// class MNASNet{
// // MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This implements the B1 variant of the model.
// public:
//     // MNASNet() {

//     // }

//     void forward(float x[in_channels][in_height][in_width], float y[in_channels][in_height][in_width]) {
//         constexpr int depths[8] = {32, 16, 24, 40, 80, 96, 192, 320};
//         // _get_depths(alpha, depths);

//         // First layer: regular conv.
//         const int l0_kernel_size = 3;
//         const int l0_stride = 2;
//         const int l0_padding = 1;
//         const int l0_groups = 1;
//         const int l0_out_channels = depths[0];
//         const int l0_out_height = conv_out_size(in_height, l0_kernel_size, l0_stride, l0_padding);
//         const int l0_out_width = conv_out_size(in_width, l0_kernel_size, l0_stride, l0_padding);
//         Conv2d<in_channels, in_height, in_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size, l0_stride, l0_padding, l0_groups> l0_conv;
//         // out_heights[0] = l0_out_height;
//         // out_widths[0] = l0_out_width;

//         const int l1_out_channels = depths[0];
//         const int l1_out_height = l0_out_height;
//         const int l1_out_width = l0_out_width;
//         BatchNorm2d<l1_out_channels, l1_out_height, l1_out_width> l1_bn;
//         // out_heights[1] = l1_out_height;
//         // out_widths[1] = l1_out_width;

//         const int l2_out_channels = depths[0];
//         const int l2_out_height = l1_out_height;
//         const int l2_out_width = l1_out_width;
//         ReLU<l2_out_channels, l2_out_height, l2_out_width> l2_relu;
//         // out_heights[2] = l2_out_height;
//         // out_widths[2] = l2_out_width;

//         // Depthwise separable, no skip.
//         const int l3_kernel_size = 3;
//         const int l3_stride = 1;
//         const int l3_padding = 1;
//         const int l3_groups = depths[0];
//         const int l3_out_channels = depths[0];
//         const int l3_out_height = conv_out_size(l2_out_height, l3_kernel_size, l3_stride, l3_padding);
//         const int l3_out_width = conv_out_size(l2_out_width, l3_kernel_size, l3_stride, l3_padding);
//         Conv2d<l2_out_channels, l2_out_height, l2_out_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size, l3_stride, l3_padding, l3_groups> l3_conv;
//         // out_heights[3] = l3_out_height;
//         // out_widths[3] = l3_out_width;

//         const int l4_out_channels = depths[0];
//         const int l4_out_height = l3_out_height;
//         const int l4_out_width = l3_out_width;
//         BatchNorm2d<l4_out_channels, l4_out_height, l4_out_width> l4_bn;
//         // out_heights[4] = l4_out_height;
//         // out_widths[4] = l4_out_width;

//         const int l5_out_channels = depths[0];
//         const int l5_out_height = l4_out_height;
//         const int l5_out_width = l4_out_width;
//         ReLU<l5_out_channels, l5_out_height, l5_out_width> l5_relu;
//         // out_heights[5] = l5_out_height;
//         // out_widths[5] = l5_out_width;

//         const int l6_kernel_size = 1;
//         const int l6_stride = 1;
//         const int l6_padding = 0;
//         const int l6_groups = 1;
//         const int l6_out_channels = depths[1];
//         const int l6_out_height = conv_out_size(l5_out_height, l6_kernel_size, l6_stride, l6_padding);
//         const int l6_out_width = conv_out_size(l5_out_width, l6_kernel_size, l6_stride, l6_padding);
//         Conv2d<l5_out_channels, l5_out_height, l5_out_width, l6_out_channels, l6_out_height, l6_out_width, l6_kernel_size, l6_stride, l6_padding, l6_groups> l6_conv;
//         // out_heights[6] = l6_out_height;
//         // out_widths[6] = l6_out_width;

//         const int l7_out_channels = depths[1];
//         const int l7_out_height = l6_out_height;
//         const int l7_out_width = l6_out_width;
//         BatchNorm2d<l7_out_channels, l7_out_height, l7_out_width> l7_bn;
//         // out_heights[7] = l7_out_height;
//         // out_widths[7] = l7_out_width;

//         // MNASNet blocks: stacks of inverted residuals.
//         const int l8_kernel_size = 3;
//         const int l8_stride = 2;
//         const int l8_expansion_factor = 3;
//         const int l8_repeats = 3;
//         const int l8_out_channels = depths[2];
//         const int l8_out_height = stack_out_size(l7_out_height, l8_kernel_size, l8_stride);
//         const int l8_out_width = stack_out_size(l7_out_width, l8_kernel_size, l8_stride);
//         _stack<l7_out_channels, l7_out_height, l7_out_width, l8_out_channels, l8_out_height, l8_out_width, l8_kernel_size, l8_stride, l8_expansion_factor, l8_repeats> l8_stack;

//         const int l9_kernel_size = 5;
//         const int l9_stride = 2;
//         const int l9_expansion_factor = 3;
//         const int l9_repeats = 3;
//         const int l9_out_channels = depths[3];
//         const int l9_out_height = stack_out_size(l8_out_height, l9_kernel_size, l9_stride);
//         const int l9_out_width = stack_out_size(l8_out_width, l9_kernel_size, l9_stride);
//         _stack<l8_out_channels, l8_out_height, l8_out_width, l9_out_channels, l9_out_height, l9_out_width, l9_kernel_size, l9_stride, l9_expansion_factor, l9_repeats> l9_stack;

//         const int l10_kernel_size = 5;
//         const int l10_stride = 2;
//         const int l10_expansion_factor = 6;
//         const int l10_repeats = 3;
//         const int l10_out_channels = depths[4];
//         const int l10_out_height = stack_out_size(l9_out_height, l10_kernel_size, l10_stride);
//         const int l10_out_width = stack_out_size(l9_out_width, l10_kernel_size, l10_stride);
//         _stack<l9_out_channels, l9_out_height, l9_out_width, l10_out_channels, l10_out_height, l10_out_width, l10_kernel_size, l10_stride, l10_expansion_factor, l10_repeats> l10_stack;

//         const int l11_kernel_size = 3;
//         const int l11_stride = 1;
//         const int l11_expansion_factor = 6;
//         const int l11_repeats = 2;
//         const int l11_out_channels = depths[5];
//         const int l11_out_height = stack_out_size(l10_out_height, l11_kernel_size, l11_stride);
//         const int l11_out_width = stack_out_size(l10_out_width, l11_kernel_size, l11_stride);
//         _stack<l10_out_channels, l10_out_height, l10_out_width, l11_out_channels, l11_out_height, l11_out_width, l11_kernel_size, l11_stride, l11_expansion_factor, l11_repeats> l11_stack;

//         const int l12_kernel_size = 5;
//         const int l12_stride = 2;
//         const int l12_expansion_factor = 6;
//         const int l12_repeats = 4;
//         const int l12_out_channels = depths[6];
//         const int l12_out_height = stack_out_size(l11_out_height, l12_kernel_size, l12_stride);
//         const int l12_out_width = stack_out_size(l11_out_width, l12_kernel_size, l12_stride);
//         _stack<l11_out_channels, l11_out_height, l11_out_width, l12_out_channels, l12_out_height, l12_out_width, l12_kernel_size, l12_stride, l12_expansion_factor, l12_repeats> l12_stack;

//         const int l13_kernel_size = 3;
//         const int l13_stride = 1;
//         const int l13_expansion_factor = 6;
//         const int l13_repeats = 1;
//         const int l13_out_channels = depths[7];
//         const int l13_out_height = stack_out_size(l12_out_height, l13_kernel_size, l13_stride);
//         const int l13_out_width = stack_out_size(l12_out_width, l13_kernel_size, l13_stride);
//         _stack<l12_out_channels, l12_out_height, l12_out_width, l13_out_channels, l13_out_height, l13_out_width, l13_kernel_size, l13_stride, l13_expansion_factor, l13_repeats> l13_stack;

//         // Final mapping to classifier input.
//         // nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
//         // nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
//         // nn.ReLU(inplace=True),

//         // const int num_classes = 1000;
//         // const float dropout = 0.2;
//         // self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(1280, num_classes))

//         float y0[l0_out_channels][l0_out_height][l0_out_width];
//         l0_conv.forward(x, y0);

//         float y1[l1_out_channels][l1_out_height][l1_out_width];
//         l1_bn.forward(y0, y1);

//         float y2[l2_out_channels][l2_out_height][l2_out_width];
//         l2_relu.forward(y1, y2);

//         float y3[l3_out_channels][l3_out_height][l3_out_width];
//         l3_conv.forward(y2, y3);

//         float y4[l4_out_channels][l4_out_height][l4_out_width];
//         l4_bn.forward(y3, y4);

//         float y5[l5_out_channels][l5_out_height][l5_out_width];
//         l5_relu.forward(y4, y5);

//         float y6[l6_out_channels][l6_out_height][l6_out_width];
//         l6_conv.forward(y5, y6);

//         float y7[l7_out_channels][l7_out_height][l7_out_width];
//         l7_bn.forward(y6, y7);

//         float y8[l8_out_channels][l8_out_height][l8_out_width];
//         l8_stack.forward(y7, y8);

//         float y9[l9_out_channels][l9_out_height][l9_out_width];
//         l9_stack.forward(y8, y9);

//         float y10[l10_out_channels][l10_out_height][l10_out_width];
//         l10_stack.forward(y9, y10);

//         float y11[l11_out_channels][l11_out_height][l11_out_width];
//         l11_stack.forward(y10, y11);

//         float y12[l12_out_channels][l12_out_height][l12_out_width];
//         l12_stack.forward(y11, y12);

//         float y13[l13_out_channels][l13_out_height][l13_out_width];
//         l13_stack.forward(y12, y13);
//     }

// // private:
//     // int out_heights[17];
//     // int out_widths[17];
// };
