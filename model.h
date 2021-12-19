#include "config.h"
#include "mnasnet.h"
// #include <Eigen/Dense>
// #include <Eigen/Core>
// #include <Eigen/LU>
// using namespace Eigen;

#define fe1_out_size(size) conv_out_size(conv_out_size(conv_out_size((size), 3, 2, 1), 3, 1, 1), 1, 1, 0)
#define fe2_out_size(size) stack_out_size(fe1_out_size((size)), 3, 2)
#define fe3_out_size(size) stack_out_size(fe2_out_size((size)), 5, 2)
#define fe4_out_size(size) stack_out_size(stack_out_size(fe3_out_size((size)), 5, 2), 3, 1)
#define fe5_out_size(size) stack_out_size(stack_out_size(fe4_out_size((size)), 5, 2), 3, 1)

#define fe1_out_channel 16
#define fe2_out_channel 24
#define fe3_out_channel 40
#define fe4_out_channel 96
#define fe5_out_channel 320


template <int in_channels, int in_height, int in_width>
class FeatureExtractor{
public:
    FeatureExtractor(const string param_path) : param_path(param_path) {}

    void forward(float x[in_channels][in_height][in_width],
                 float layer1[fe1_out_channel][fe1_out_size(in_height)][fe1_out_size(in_width)],
                 float layer2[fe2_out_channel][fe2_out_size(in_height)][fe2_out_size(in_width)],
                 float layer3[fe3_out_channel][fe3_out_size(in_height)][fe3_out_size(in_width)],
                 float layer4[fe4_out_channel][fe4_out_size(in_height)][fe4_out_size(in_width)],
                 float layer5[fe5_out_channel][fe5_out_size(in_height)][fe5_out_size(in_width)]) {

        constexpr int depths[8] = {32, 16, 24, 40, 80, 96, 192, 320};

        // First layer: regular conv.
        const int l0_kernel_size = 3;
        const int l0_stride = 2;
        const int l0_padding = 1;
        const int l0_groups = 1;
        const int l0_out_channels = depths[0];
        const int l0_out_height = conv_out_size(in_height, l0_kernel_size, l0_stride, l0_padding);
        const int l0_out_width = conv_out_size(in_width, l0_kernel_size, l0_stride, l0_padding);
        Conv2d<in_channels, in_height, in_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size, l0_stride, l0_padding, l0_groups> l0_conv(param_path + "/layer1.0");

        const int l1_out_channels = depths[0];
        const int l1_out_height = l0_out_height;
        const int l1_out_width = l0_out_width;
        BatchNorm2d<l1_out_channels, l1_out_height, l1_out_width> l1_bn(param_path + "/layer1.1");

        const int l2_out_channels = depths[0];
        const int l2_out_height = l1_out_height;
        const int l2_out_width = l1_out_width;
        ReLU<l2_out_channels, l2_out_height, l2_out_width> l2_relu;

        // Depthwise separable, no skip.
        const int l3_kernel_size = 3;
        const int l3_stride = 1;
        const int l3_padding = 1;
        const int l3_groups = depths[0];
        const int l3_out_channels = depths[0];
        const int l3_out_height = conv_out_size(l2_out_height, l3_kernel_size, l3_stride, l3_padding);
        const int l3_out_width = conv_out_size(l2_out_width, l3_kernel_size, l3_stride, l3_padding);
        Conv2d<l2_out_channels, l2_out_height, l2_out_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size, l3_stride, l3_padding, l3_groups> l3_conv(param_path + "/layer1.3");

        const int l4_out_channels = depths[0];
        const int l4_out_height = l3_out_height;
        const int l4_out_width = l3_out_width;
        BatchNorm2d<l4_out_channels, l4_out_height, l4_out_width> l4_bn(param_path + "/layer1.4");

        const int l5_out_channels = depths[0];
        const int l5_out_height = l4_out_height;
        const int l5_out_width = l4_out_width;
        ReLU<l5_out_channels, l5_out_height, l5_out_width> l5_relu;

        const int l6_kernel_size = 1;
        const int l6_stride = 1;
        const int l6_padding = 0;
        const int l6_groups = 1;
        const int l6_out_channels = depths[1];
        const int l6_out_height = conv_out_size(l5_out_height, l6_kernel_size, l6_stride, l6_padding);
        const int l6_out_width = conv_out_size(l5_out_width, l6_kernel_size, l6_stride, l6_padding);
        Conv2d<l5_out_channels, l5_out_height, l5_out_width, l6_out_channels, l6_out_height, l6_out_width, l6_kernel_size, l6_stride, l6_padding, l6_groups> l6_conv(param_path + "/layer1.6");

        const int l7_out_channels = depths[1];
        const int l7_out_height = l6_out_height;
        const int l7_out_width = l6_out_width;
        BatchNorm2d<l7_out_channels, l7_out_height, l7_out_width> l7_bn(param_path + "/layer1.7");

        // MNASNet blocks: stacks of inverted residuals.
        const int l8_kernel_size = 3;
        const int l8_stride = 2;
        const int l8_expansion_factor = 3;
        const int l8_repeats = 3;
        const int l8_out_channels = depths[2];
        const int l8_out_height = stack_out_size(l7_out_height, l8_kernel_size, l8_stride);
        const int l8_out_width = stack_out_size(l7_out_width, l8_kernel_size, l8_stride);
        _stack<l7_out_channels, l7_out_height, l7_out_width, l8_out_channels, l8_out_height, l8_out_width, l8_kernel_size, l8_stride, l8_expansion_factor, l8_repeats> l8_stack(param_path + "/layer2.0");

        const int l9_kernel_size = 5;
        const int l9_stride = 2;
        const int l9_expansion_factor = 3;
        const int l9_repeats = 3;
        const int l9_out_channels = depths[3];
        const int l9_out_height = stack_out_size(l8_out_height, l9_kernel_size, l9_stride);
        const int l9_out_width = stack_out_size(l8_out_width, l9_kernel_size, l9_stride);
        _stack<l8_out_channels, l8_out_height, l8_out_width, l9_out_channels, l9_out_height, l9_out_width, l9_kernel_size, l9_stride, l9_expansion_factor, l9_repeats> l9_stack(param_path + "/layer3.0");

        const int l10_kernel_size = 5;
        const int l10_stride = 2;
        const int l10_expansion_factor = 6;
        const int l10_repeats = 3;
        const int l10_out_channels = depths[4];
        const int l10_out_height = stack_out_size(l9_out_height, l10_kernel_size, l10_stride);
        const int l10_out_width = stack_out_size(l9_out_width, l10_kernel_size, l10_stride);
        _stack<l9_out_channels, l9_out_height, l9_out_width, l10_out_channels, l10_out_height, l10_out_width, l10_kernel_size, l10_stride, l10_expansion_factor, l10_repeats> l10_stack(param_path + "/layer4.0");

        const int l11_kernel_size = 3;
        const int l11_stride = 1;
        const int l11_expansion_factor = 6;
        const int l11_repeats = 2;
        const int l11_out_channels = depths[5];
        const int l11_out_height = stack_out_size(l10_out_height, l11_kernel_size, l11_stride);
        const int l11_out_width = stack_out_size(l10_out_width, l11_kernel_size, l11_stride);
        _stack<l10_out_channels, l10_out_height, l10_out_width, l11_out_channels, l11_out_height, l11_out_width, l11_kernel_size, l11_stride, l11_expansion_factor, l11_repeats> l11_stack(param_path + "/layer4.1");

        const int l12_kernel_size = 5;
        const int l12_stride = 2;
        const int l12_expansion_factor = 6;
        const int l12_repeats = 4;
        const int l12_out_channels = depths[6];
        const int l12_out_height = stack_out_size(l11_out_height, l12_kernel_size, l12_stride);
        const int l12_out_width = stack_out_size(l11_out_width, l12_kernel_size, l12_stride);
        _stack<l11_out_channels, l11_out_height, l11_out_width, l12_out_channels, l12_out_height, l12_out_width, l12_kernel_size, l12_stride, l12_expansion_factor, l12_repeats> l12_stack(param_path + "/layer5.0");

        const int l13_kernel_size = 3;
        const int l13_stride = 1;
        const int l13_expansion_factor = 6;
        const int l13_repeats = 1;
        const int l13_out_channels = depths[7];
        const int l13_out_height = stack_out_size(l12_out_height, l13_kernel_size, l13_stride);
        const int l13_out_width = stack_out_size(l12_out_width, l13_kernel_size, l13_stride);
        _stack<l12_out_channels, l12_out_height, l12_out_width, l13_out_channels, l13_out_height, l13_out_width, l13_kernel_size, l13_stride, l13_expansion_factor, l13_repeats> l13_stack(param_path + "/layer5.1");


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

        float y8[l8_out_channels][l8_out_height][l8_out_width];
        l8_stack.forward(y7, y8);

        float y9[l9_out_channels][l9_out_height][l9_out_width];
        l9_stack.forward(y8, y9);

        float y10[l10_out_channels][l10_out_height][l10_out_width];
        l10_stack.forward(y9, y10);

        float y11[l11_out_channels][l11_out_height][l11_out_width];
        l11_stack.forward(y10, y11);

        float y12[l12_out_channels][l12_out_height][l12_out_width];
        l12_stack.forward(y11, y12);

        float y13[l13_out_channels][l13_out_height][l13_out_width];
        l13_stack.forward(y12, y13);


        for (int i = 0; i < fe1_out_channel; i++) for (int j = 0; j < fe1_out_size(in_height); j++) for (int k = 0; k < fe1_out_size(in_width); k++)
            layer1[i][j][k] = y7[i][j][k];

        for (int i = 0; i < fe2_out_channel; i++) for (int j = 0; j < fe2_out_size(in_height); j++) for (int k = 0; k < fe2_out_size(in_width); k++)
            layer2[i][j][k] = y8[i][j][k];

        for (int i = 0; i < fe3_out_channel; i++) for (int j = 0; j < fe3_out_size(in_height); j++) for (int k = 0; k < fe3_out_size(in_width); k++)
            layer3[i][j][k] = y9[i][j][k];

        for (int i = 0; i < fe4_out_channel; i++) for (int j = 0; j < fe4_out_size(in_height); j++) for (int k = 0; k < fe4_out_size(in_width); k++)
            layer4[i][j][k] = y11[i][j][k];

        for (int i = 0; i < fe5_out_channel; i++) for (int j = 0; j < fe5_out_size(in_height); j++) for (int k = 0; k < fe5_out_size(in_width); k++)
            layer5[i][j][k] = y13[i][j][k];

    }

private:
    string param_path;

};

