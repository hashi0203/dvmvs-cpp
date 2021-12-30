#pragma once
#include "config.h"
#include "functional.h"
#include "layers.h"
#include "mnasnet.h"

template <int channels, int height, int width, int kernel_size, bool apply_bn_relu>
class StandardLayer{
public:
    StandardLayer(const string param_path) : param_path(param_path) {}

    void forward(const float x[channels][height][width], float y[channels][height][width]) {
        const int stride = 1;

        const bool l0_apply_bn_relu = true;
        conv_layer<channels, height, width, channels, height, width, kernel_size, stride, l0_apply_bn_relu> l0_conv_layer(param_path + ".conv1");
        conv_layer<channels, height, width, channels, height, width, kernel_size, stride, apply_bn_relu> l1_conv_layer(param_path + ".conv2");

        float y0[channels][height][width];
        l0_conv_layer.forward(x, y0);

        // float y1[channels][height][width];
        l1_conv_layer.forward(y0, y);

        // for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
        //     y[i][j][k] = y1[i][j][k];
    }

private:
    string param_path;
};


template <int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int kernel_size>
class DownconvolutionLayer{
public:
    DownconvolutionLayer(const string param_path) : param_path(param_path) {}

    void forward(const float x[in_channels][in_height][in_width], float y[out_channels][out_height][out_width]) {
        const int stride = 2;
        const bool apply_bn_relu = true;

        conv_layer<in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, apply_bn_relu> l0_down_conv(param_path + ".down_conv");

        // float y0[out_channels][out_height][out_width];
        l0_down_conv.forward(x, y);

        // for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
        //     y[i][j][k] = y0[i][j][k];
    }

private:
    string param_path;
};


template <int in_channels, int in_height, int in_width, int out_channels, int kernel_size>
class UpconvolutionLayer{
public:
    UpconvolutionLayer(const string param_path) : param_path(param_path) {}

    void forward(const float x[in_channels][in_height][in_width], float y[out_channels][in_height * 2][in_width * 2]) {
        const int out_height = in_height * 2;
        const int out_width = in_width * 2;
        float up_x[in_channels][out_height][out_width];
        interpolate<in_channels, in_height, in_width, out_height, out_width>(x, up_x, "bilinear");

        const int stride = 1;
        const bool apply_bn_relu = true;

        conv_layer<in_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, apply_bn_relu> l0_conv_layer(param_path + ".conv");

        // float y0[out_channels][out_height][out_width];
        l0_conv_layer.forward(up_x, y);

        // for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
        //     y[i][j][k] = y0[i][j][k];
    }

private:
    string param_path;
};

template <int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int kernel_size>
class EncoderBlock{
public:
    EncoderBlock(const string param_path) : param_path(param_path) {}

    void forward(const float x[in_channels][in_height][in_width], float y[out_channels][out_height][out_width]) {

        DownconvolutionLayer<in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size> l0_down_convolution(param_path + ".down_convolution");
        float y0[out_channels][out_height][out_width];
        // float ***y0 = new float**[out_channels];
        // new_3d(y0, out_channels, out_height, out_width);
        l0_down_convolution.forward(x, y0);

        const bool apply_bn_relu = true;
        StandardLayer<out_channels, out_height, out_width, kernel_size, apply_bn_relu> l1_standard_convolution(param_path + ".standard_convolution");
        // float y1[out_channels][out_height][out_width];
        l1_standard_convolution.forward(y0, y);
        // delete_3d(y0, out_channels, out_height, out_width);

        // for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
        //     y[i][j][k] = y1[i][j][k];
    }

private:
    string param_path;
};

template <int in_channels, int in_height, int in_width, int kernel_size, bool apply_bn_relu, bool plus_one>
class DecoderBlock{
public:
    DecoderBlock(const string param_path) : param_path(param_path) {}

    void forward(const float x[in_channels][in_height][in_width],
                 const float skip[in_channels / 2][in_height * 2][in_width * 2],
                 const float depth[1][in_height][in_width],
                 float y[in_channels / 2][in_height * 2][in_width * 2]) {

        const int out_height = in_height * 2;
        const int out_width = in_width * 2;
        const int out_channels = in_channels / 2;

        UpconvolutionLayer<in_channels, in_height, in_width, out_channels, kernel_size> l0_up_convolution(param_path + ".up_convolution");
        float y0[out_channels][out_height][out_width];
        l0_up_convolution.forward(x, y0);

        const int stride = 1;

        // Aggregate skip and upsampled input
        const int l1_in_channels = plus_one ? in_channels +  1 : in_channels;

        float x1[l1_in_channels][out_height][out_width];
        for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
            x1[i][j][k] = y0[i][j][k];
        for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
            x1[i+out_channels][j][k] = skip[i][j][k];
        if (plus_one) {
            float up_depth[1][out_height][out_width];
            interpolate<1, in_height, in_width, out_height, out_width>(depth, up_depth, "bilinear");
            for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
                x1[in_channels][j][k] = up_depth[0][j][k];
        }

        const bool l1_apply_bn_relu = true;
        conv_layer<l1_in_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, l1_apply_bn_relu> l1_conv_layer(param_path + ".convolution1");
        float y1[out_channels][out_height][out_width];
        l1_conv_layer.forward(x1, y1);

        // Learn from aggregation
        const bool l2_apply_bn_relu = apply_bn_relu;
        conv_layer<out_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, l2_apply_bn_relu> l2_conv_layer(param_path + ".convolution2");
        l1_conv_layer.forward(y1, y);
    }

private:
    string param_path;
};



template <int in_channels, int in_height, int in_width>
class FeatureExtractor{
public:
    FeatureExtractor(const string param_path) : param_path(param_path) {}

    void forward(const float x[in_channels][in_height][in_width],
                 float layer1[fe1_out_channels][fe1_out_size(in_height)][fe1_out_size(in_width)],
                 float layer2[fe2_out_channels][fe2_out_size(in_height)][fe2_out_size(in_width)],
                 float layer3[fe3_out_channels][fe3_out_size(in_height)][fe3_out_size(in_width)],
                 float layer4[fe4_out_channels][fe4_out_size(in_height)][fe4_out_size(in_width)],
                 float layer5[fe5_out_channels][fe5_out_size(in_height)][fe5_out_size(in_width)]) {

        constexpr int depths[8] = {32, fe1_out_channels, fe2_out_channels, fe3_out_channels, 80, fe4_out_channels, 192, fe5_out_channels};

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

        // float y1[l1_out_channels][l1_out_height][l1_out_width];
        l1_bn.forward(y0, y0);

        // float y2[l2_out_channels][l2_out_height][l2_out_width];
        l2_relu.forward(y0, y0);

        float y3[l3_out_channels][l3_out_height][l3_out_width];
        l3_conv.forward(y0, y3);

        // float y4[l4_out_channels][l4_out_height][l4_out_width];
        l4_bn.forward(y3, y3);

        // float y5[l5_out_channels][l5_out_height][l5_out_width];
        l5_relu.forward(y3, y3);

        // float y6[l6_out_channels][l6_out_height][l6_out_width];
        l6_conv.forward(y3, layer1);

        // float y7[l7_out_channels][l7_out_height][l7_out_width];
        l7_bn.forward(layer1, layer1);

        // float y8[l8_out_channels][l8_out_height][l8_out_width];
        l8_stack.forward(layer1, layer2);

        // float y9[l9_out_channels][l9_out_height][l9_out_width];
        l9_stack.forward(layer2, layer3);

        float y10[l10_out_channels][l10_out_height][l10_out_width];
        l10_stack.forward(layer3, y10);

        // float y11[l11_out_channels][l11_out_height][l11_out_width];
        l11_stack.forward(y10, layer4);

        float y12[l12_out_channels][l12_out_height][l12_out_width];
        l12_stack.forward(layer4, y12);

        // float y13[l13_out_channels][l13_out_height][l13_out_width];
        l13_stack.forward(y12, layer5);


        // for (int i = 0; i < fe1_out_channels; i++) for (int j = 0; j < fe1_out_size(in_height); j++) for (int k = 0; k < fe1_out_size(in_width); k++)
        //     layer1[i][j][k] = y7[i][j][k];

        // for (int i = 0; i < fe2_out_channels; i++) for (int j = 0; j < fe2_out_size(in_height); j++) for (int k = 0; k < fe2_out_size(in_width); k++)
        //     layer2[i][j][k] = y8[i][j][k];

        // for (int i = 0; i < fe3_out_channels; i++) for (int j = 0; j < fe3_out_size(in_height); j++) for (int k = 0; k < fe3_out_size(in_width); k++)
        //     layer3[i][j][k] = y9[i][j][k];

        // for (int i = 0; i < fe4_out_channels; i++) for (int j = 0; j < fe4_out_size(in_height); j++) for (int k = 0; k < fe4_out_size(in_width); k++)
        //     layer4[i][j][k] = y11[i][j][k];

        // for (int i = 0; i < fe5_out_channels; i++) for (int j = 0; j < fe5_out_size(in_height); j++) for (int k = 0; k < fe5_out_size(in_width); k++)
        //     layer5[i][j][k] = y13[i][j][k];

    }

private:
    string param_path;
};


template <int in_height, int in_width>
class FeatureShrinker{
// Module that adds a FPN from on top of a set of feature maps. This is based on
// `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
// The feature maps are currently supposed to be in increasing depth order.
// The input to the model is expected to be an OrderedDict[Tensor], containing
// the feature maps on top of which the FPN will be added.
public:
    FeatureShrinker(const string param_path) : param_path(param_path) {}

    void forward(const float layer1[fe1_out_channels][fe1_out_size(in_height)][fe1_out_size(in_width)],
                 const float layer2[fe2_out_channels][fe2_out_size(in_height)][fe2_out_size(in_width)],
                 const float layer3[fe3_out_channels][fe3_out_size(in_height)][fe3_out_size(in_width)],
                 const float layer4[fe4_out_channels][fe4_out_size(in_height)][fe4_out_size(in_width)],
                 const float layer5[fe5_out_channels][fe5_out_size(in_height)][fe5_out_size(in_width)],
                 float features_half[fpn_output_channels][fe1_out_size(in_height)][fe1_out_size(in_width)],
                 float features_quarter[fpn_output_channels][fe2_out_size(in_height)][fe2_out_size(in_width)],
                 float features_one_eight[fpn_output_channels][fe3_out_size(in_height)][fe3_out_size(in_width)],
                 float features_one_sixteen[fpn_output_channels][fe4_out_size(in_height)][fe4_out_size(in_width)]) {

        const int stride = 1;
        const int groups = 1;

        const int inner_kernel_size = 1;
        const int inner_padding = 0;
        const int layer_kernel_size = 3;
        const int layer_padding = 1;

        // layer5
        Conv2d<fe5_out_channels, fe5_out_size(in_height), fe5_out_size(in_width), fpn_output_channels, fe5_out_size(in_height), fe5_out_size(in_width), inner_kernel_size, stride, inner_padding, groups> i5_conv(param_path + "/fpn.inner_blocks.4");
        float inner5[fpn_output_channels][fe5_out_size(in_height)][fe5_out_size(in_width)];
        i5_conv.forward(layer5, inner5);

        Conv2d<fpn_output_channels, fe5_out_size(in_height), fe5_out_size(in_width), fpn_output_channels, fe5_out_size(in_height), fe5_out_size(in_width), layer_kernel_size, stride, layer_padding, groups> l5_conv(param_path + "/fpn.layer_blocks.4");
        float features_smallest[fpn_output_channels][fe5_out_size(in_height)][fe5_out_size(in_width)];
        l5_conv.forward(inner5, features_smallest);


        // layer4
        Conv2d<fe4_out_channels, fe4_out_size(in_height), fe4_out_size(in_width), fpn_output_channels, fe4_out_size(in_height), fe4_out_size(in_width), inner_kernel_size, stride, inner_padding, groups> i4_conv(param_path + "/fpn.inner_blocks.3");
        float inner4[fpn_output_channels][fe4_out_size(in_height)][fe4_out_size(in_width)];
        i4_conv.forward(layer4, inner4);

        float top_down4[fpn_output_channels][fe4_out_size(in_height)][fe4_out_size(in_width)];
        interpolate<fpn_output_channels, fe5_out_size(in_height), fe5_out_size(in_width), fe4_out_size(in_height), fe4_out_size(in_width)>(inner5, top_down4);
        for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < fe4_out_size(in_height); j++) for (int k = 0; k < fe4_out_size(in_width); k++)
            inner4[i][j][k] += top_down4[i][j][k];

        Conv2d<fpn_output_channels, fe4_out_size(in_height), fe4_out_size(in_width), fpn_output_channels, fe4_out_size(in_height), fe4_out_size(in_width), layer_kernel_size, stride, layer_padding, groups> l4_conv(param_path + "/fpn.layer_blocks.3");
        l4_conv.forward(inner4, features_one_sixteen);


        // layer3
        Conv2d<fe3_out_channels, fe3_out_size(in_height), fe3_out_size(in_width), fpn_output_channels, fe3_out_size(in_height), fe3_out_size(in_width), inner_kernel_size, stride, inner_padding, groups> i3_conv(param_path + "/fpn.inner_blocks.2");
        float inner3[fpn_output_channels][fe3_out_size(in_height)][fe3_out_size(in_width)];
        i3_conv.forward(layer3, inner3);

        float top_down3[fpn_output_channels][fe3_out_size(in_height)][fe3_out_size(in_width)];
        interpolate<fpn_output_channels, fe4_out_size(in_height), fe4_out_size(in_width), fe3_out_size(in_height), fe3_out_size(in_width)>(inner4, top_down3);
        for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < fe3_out_size(in_height); j++) for (int k = 0; k < fe3_out_size(in_width); k++)
            inner3[i][j][k] += top_down3[i][j][k];

        Conv2d<fpn_output_channels, fe3_out_size(in_height), fe3_out_size(in_width), fpn_output_channels, fe3_out_size(in_height), fe3_out_size(in_width), layer_kernel_size, stride, layer_padding, groups> l3_conv(param_path + "/fpn.layer_blocks.2");
        l3_conv.forward(inner3, features_one_eight);


        // layer2
        Conv2d<fe2_out_channels, fe2_out_size(in_height), fe2_out_size(in_width), fpn_output_channels, fe2_out_size(in_height), fe2_out_size(in_width), inner_kernel_size, stride, inner_padding, groups> i2_conv(param_path + "/fpn.inner_blocks.1");
        float inner2[fpn_output_channels][fe2_out_size(in_height)][fe2_out_size(in_width)];
        i2_conv.forward(layer2, inner2);

        float top_down2[fpn_output_channels][fe2_out_size(in_height)][fe2_out_size(in_width)];
        interpolate<fpn_output_channels, fe3_out_size(in_height), fe3_out_size(in_width), fe2_out_size(in_height), fe2_out_size(in_width)>(inner3, top_down2);
        for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < fe2_out_size(in_height); j++) for (int k = 0; k < fe2_out_size(in_width); k++)
            inner2[i][j][k] += top_down2[i][j][k];

        Conv2d<fpn_output_channels, fe2_out_size(in_height), fe2_out_size(in_width), fpn_output_channels, fe2_out_size(in_height), fe2_out_size(in_width), layer_kernel_size, stride, layer_padding, groups> l2_conv(param_path + "/fpn.layer_blocks.1");
        l2_conv.forward(inner2, features_quarter);


        // layer1
        Conv2d<fe1_out_channels, fe1_out_size(in_height), fe1_out_size(in_width), fpn_output_channels, fe1_out_size(in_height), fe1_out_size(in_width), inner_kernel_size, stride, inner_padding, groups> i1_conv(param_path + "/fpn.inner_blocks.0");
        float inner1[fpn_output_channels][fe1_out_size(in_height)][fe1_out_size(in_width)];
        i1_conv.forward(layer1, inner1);

        float top_down1[fpn_output_channels][fe1_out_size(in_height)][fe1_out_size(in_width)];
        interpolate<fpn_output_channels, fe2_out_size(in_height), fe2_out_size(in_width), fe1_out_size(in_height), fe1_out_size(in_width)>(inner2, top_down1);
        for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < fe1_out_size(in_height); j++) for (int k = 0; k < fe1_out_size(in_width); k++)
            inner1[i][j][k] += top_down1[i][j][k];

        Conv2d<fpn_output_channels, fe1_out_size(in_height), fe1_out_size(in_width), fpn_output_channels, fe1_out_size(in_height), fe1_out_size(in_width), layer_kernel_size, stride, layer_padding, groups> l1_conv(param_path + "/fpn.layer_blocks.0");
        l1_conv.forward(inner1, features_half);
    }

private:
    string param_path;
};


template <int in_height, int in_width>
class CostVolumeEncoder{
public:
    CostVolumeEncoder(const string param_path) : param_path(param_path) {}

    void forward(const float features_half[fe1_out_channels][fe1_out_size(in_height)][fe1_out_size(in_width)],
                 const float features_quarter[fe2_out_channels][fe2_out_size(in_height)][fe2_out_size(in_width)],
                 const float features_one_eight[fe3_out_channels][fe3_out_size(in_height)][fe3_out_size(in_width)],
                 const float features_one_sixteen[fe4_out_channels][fe4_out_size(in_height)][fe4_out_size(in_width)],
                 const float cost_volume[n_depth_levels][fe1_out_size(in_height)][fe1_out_size(in_width)],
                 float skip0[hyper_channels][fe1_out_size(in_height)][fe1_out_size(in_width)],
                 float skip1[hyper_channels * 2][fe2_out_size(in_height)][fe2_out_size(in_width)],
                 float skip2[hyper_channels * 4][fe3_out_size(in_height)][fe3_out_size(in_width)],
                 float skip3[hyper_channels * 8][fe4_out_size(in_height)][fe4_out_size(in_width)],
                 float bottom[hyper_channels * 16][fe5_out_size(in_height)][fe5_out_size(in_width)]) {

        const int stride = 1;
        const bool apply_bn_relu = true;


        // 1st set
        const int l0_kernel_size = 5;
        const int l0_in_channels = n_depth_levels + fpn_output_channels;

        const int l0_mid_channels = hyper_channels;
        const int l0_mid_height = fe1_out_size(in_height);
        const int l0_mid_width = fe1_out_size(in_width);

        const int l0_out_channels = hyper_channels * 2;
        const int l0_out_height = fe2_out_size(in_height);
        const int l0_out_width = fe2_out_size(in_width);

        conv_layer<l0_in_channels, l0_mid_height, l0_mid_width, l0_mid_channels, l0_mid_height, l0_mid_width, l0_kernel_size, stride, apply_bn_relu> l0_aggregator(param_path + "/aggregator0");
        EncoderBlock<l0_mid_channels, l0_mid_height, l0_mid_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size> l0_encoder_block(param_path + "/encoder_block0");

        float l0_in[l0_in_channels][fe1_out_size(in_height)][fe1_out_size(in_width)];
        for (int i = 0; i < n_depth_levels; i++) for (int j = 0; j < l0_mid_height; j++) for (int k = 0; k < l0_mid_width; k++)
            l0_in[i][j][k] = features_half[i][j][k];
        for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < l0_mid_height; j++) for (int k = 0; k < l0_mid_width; k++)
            l0_in[i+n_depth_levels][j][k] = cost_volume[i][j][k];
        l0_aggregator.forward(l0_in, skip0);

        float l0_out[l0_out_channels][l0_out_height][l0_out_width];
        l0_encoder_block.forward(skip0, l0_out);


        // 2nd set
        const int l1_kernel_size = 3;
        const int l1_in_channels = l0_out_channels + fpn_output_channels;

        const int l1_mid_channels = hyper_channels * 2;
        const int l1_mid_height = fe2_out_size(in_height);
        const int l1_mid_width = fe2_out_size(in_width);

        const int l1_out_channels = hyper_channels * 4;
        const int l1_out_height = fe3_out_size(in_height);
        const int l1_out_width = fe3_out_size(in_width);

        conv_layer<l1_in_channels, l1_mid_height, l1_mid_width, l1_mid_channels, l1_mid_height, l1_mid_width, l1_kernel_size, stride, apply_bn_relu> l1_aggregator(param_path + "/aggregator1");
        EncoderBlock<l1_mid_channels, l1_mid_height, l1_mid_width, l1_out_channels, l1_out_height, l1_out_width, l1_kernel_size> l1_encoder_block(param_path + "/encoder_block1");

        float l1_in[l1_in_channels][fe2_out_size(in_height)][fe2_out_size(in_width)];
        for (int i = 0; i < l0_out_channels; i++) for (int j = 0; j < l1_mid_height; j++) for (int k = 0; k < l1_mid_width; k++)
            l1_in[i][j][k] = features_quarter[i][j][k];
        for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < l1_mid_height; j++) for (int k = 0; k < l1_mid_width; k++)
            l1_in[i+l0_out_channels][j][k] = l0_out[i][j][k];
        l1_aggregator.forward(l1_in, skip1);

        float l1_out[l1_out_channels][l1_out_height][l1_out_width];
        l1_encoder_block.forward(skip1, l1_out);


        // 3rd set
        const int l2_kernel_size = 3;
        const int l2_in_channels = l1_out_channels + fpn_output_channels;

        const int l2_mid_channels = hyper_channels * 4;
        const int l2_mid_height = fe3_out_size(in_height);
        const int l2_mid_width = fe3_out_size(in_width);

        const int l2_out_channels = hyper_channels * 8;
        const int l2_out_height = fe4_out_size(in_height);
        const int l2_out_width = fe4_out_size(in_width);

        conv_layer<l2_in_channels, l2_mid_height, l2_mid_width, l2_mid_channels, l2_mid_height, l2_mid_width, l2_kernel_size, stride, apply_bn_relu> l2_aggregator(param_path + "/aggregator2");
        EncoderBlock<l2_mid_channels, l2_mid_height, l2_mid_width, l2_out_channels, l2_out_height, l2_out_width, l2_kernel_size> l2_encoder_block(param_path + "/encoder_block2");

        float l2_in[l2_in_channels][fe3_out_size(in_height)][fe3_out_size(in_width)];
        for (int i = 0; i < l1_out_channels; i++) for (int j = 0; j < l2_mid_height; j++) for (int k = 0; k < l2_mid_width; k++)
            l2_in[i][j][k] = features_one_eight[i][j][k];
        for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < l2_mid_height; j++) for (int k = 0; k < l2_mid_width; k++)
            l2_in[i+l1_out_channels][j][k] = l1_out[i][j][k];
        l2_aggregator.forward(l2_in, skip2);

        float l2_out[l2_out_channels][l2_out_height][l2_out_width];
        l2_encoder_block.forward(skip2, l2_out);


        // 4th set
        const int l3_kernel_size = 3;
        const int l3_in_channels = l2_out_channels + fpn_output_channels;

        const int l3_mid_channels = hyper_channels * 8;
        const int l3_mid_height = fe4_out_size(in_height);
        const int l3_mid_width = fe4_out_size(in_width);

        const int l3_out_channels = hyper_channels * 16;
        const int l3_out_height = fe5_out_size(in_height);
        const int l3_out_width = fe5_out_size(in_width);

        conv_layer<l3_in_channels, l3_mid_height, l3_mid_width, l3_mid_channels, l3_mid_height, l3_mid_width, l3_kernel_size, stride, apply_bn_relu> l3_aggregator(param_path + "/aggregator3");
        EncoderBlock<l3_mid_channels, l3_mid_height, l3_mid_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size> l3_encoder_block(param_path + "/encoder_block3");

        float l3_in[l3_in_channels][fe4_out_size(in_height)][fe4_out_size(in_width)];
        for (int i = 0; i < l2_out_channels; i++) for (int j = 0; j < l3_mid_height; j++) for (int k = 0; k < l3_mid_width; k++)
            l3_in[i][j][k] = features_one_sixteen[i][j][k];
        for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < l3_mid_height; j++) for (int k = 0; k < l3_mid_width; k++)
            l3_in[i+l2_out_channels][j][k] = l2_out[i][j][k];
        l3_aggregator.forward(l3_in, skip3);

        // float l3_out[l3_out_channels][l3_out_height][l3_out_width];
        l3_encoder_block.forward(skip3, bottom);

        // for (int i = 0; i < l3_out_channels; i++) for (int j = 0; j < l3_out_height; j++) for (int k = 0; k < l3_out_width; k++)
        //     bottom[i][j][k] = l3_out[i][j][k];

    }

private:
    string param_path;
};


template <int in_height, int in_width>
class CostVolumeDecoder{
public:
    CostVolumeDecoder(const string param_path) : param_path(param_path) {}

    void forward(const float image[3][in_height][in_width],
                 const float skip0[hyper_channels][fe1_out_size(in_height)][fe1_out_size(in_width)],
                 const float skip1[hyper_channels * 2][fe2_out_size(in_height)][fe2_out_size(in_width)],
                 const float skip2[hyper_channels * 4][fe3_out_size(in_height)][fe3_out_size(in_width)],
                 const float skip3[hyper_channels * 8][fe4_out_size(in_height)][fe4_out_size(in_width)],
                 const float bottom[hyper_channels * 16][fe5_out_size(in_height)][fe5_out_size(in_width)],
                 float depth_full[in_height][in_width]) {
                //  float depth_full[in_height][in_width],
                //  float depth_half[fe1_out_size(in_height)][fe1_out_size(in_width)],
                //  float depth_quarter[fe2_out_size(in_height)][fe2_out_size(in_width)],
                //  float depth_one_eight[fe3_out_size(in_height)][fe3_out_size(in_width)],
                //  float depth_one_sixteen[fe4_out_size(in_height)][fe4_out_size(in_width)]) {


        const float inverse_depth_base = 1 / max_depth;
        const float inverse_depth_multiplier = 1 / min_depth - 1 / max_depth;

        const bool apply_bn_relu = true;

        // 1st set
        const int l0_in_channels = hyper_channels * 16;
        const int l0_in_height = fe5_out_size(in_height);
        const int l0_in_width = fe5_out_size(in_width);

        const int l0_kernel_size = 3;
        const bool l0_plus_one = false;
        DecoderBlock<l0_in_channels, l0_in_height, l0_in_width, l0_kernel_size, apply_bn_relu, l0_plus_one> l0_decoder_block(param_path + "/decoder_block1");

        const int l0_out_channels = hyper_channels * 8;
        const int l0_out_height = fe4_out_size(in_height);
        const int l0_out_width = fe4_out_size(in_width);
        depth_layer_3x3<l0_out_channels, l0_out_height, l0_out_width> l0_depth_layer_one_sixteen(param_path + "/depth_layer_one_sixteen");

        float null_depth[1][l0_in_height][l0_in_width];
        float decoder_block1[l0_out_channels][l0_out_height][l0_out_width];
        l0_decoder_block.forward(bottom, skip3, null_depth, decoder_block1);

        float sigmoid_depth_one_sixteen[1][l0_out_height][l0_out_width];
        l0_depth_layer_one_sixteen.forward(decoder_block1, sigmoid_depth_one_sixteen);

        // float depth_one_sixteen[l0_out_height][l0_out_width];
        // for (int j = 0; j < l0_out_height; j++) for (int k = 0; k < l0_out_width; k++)
        //     depth_one_sixteen[j][k] = 1.0 / (inverse_depth_multiplier * sigmoid_depth_one_sixteen[0][j][k] + inverse_depth_base);


        // 2nd set
        const int l1_kernel_size = 3;
        const bool l1_plus_one = true;
        DecoderBlock<l0_out_channels, l0_out_height, l0_out_width, l1_kernel_size, apply_bn_relu, l1_plus_one> l1_decoder_block(param_path + "/decoder_block2");

        const int l1_out_channels = hyper_channels * 4;
        const int l1_out_height = fe3_out_size(in_height);
        const int l1_out_width = fe3_out_size(in_width);
        depth_layer_3x3<l1_out_channels, l1_out_height, l1_out_width> l1_depth_layer_one_eight(param_path + "/depth_layer_one_eight");

        float decoder_block2[l1_out_channels][l1_out_height][l1_out_width];
        l1_decoder_block.forward(decoder_block1, skip2, sigmoid_depth_one_sixteen, decoder_block2);

        float sigmoid_depth_one_eight[1][l1_out_height][l1_out_width];
        l1_depth_layer_one_eight.forward(decoder_block2, sigmoid_depth_one_eight);

        // float depth_one_eight[l1_out_height][l1_out_width];
        // for (int j = 0; j < l1_out_height; j++) for (int k = 0; k < l1_out_width; k++)
        //     depth_one_eight[j][k] = 1.0 / (inverse_depth_multiplier * sigmoid_depth_one_eight[0][j][k] + inverse_depth_base);


        // 3rd set
        const int l2_kernel_size = 3;
        const bool l2_plus_one = true;
        DecoderBlock<l1_out_channels, l1_out_height, l1_out_width, l2_kernel_size, apply_bn_relu, l2_plus_one> l2_decoder_block(param_path + "/decoder_block3");

        const int l2_out_channels = hyper_channels * 2;
        const int l2_out_height = fe2_out_size(in_height);
        const int l2_out_width = fe2_out_size(in_width);
        depth_layer_3x3<l2_out_channels, l2_out_height, l2_out_width> l2_depth_layer_quarter(param_path + "/depth_layer_quarter");

        float decoder_block3[l2_out_channels][l2_out_height][l2_out_width];
        l2_decoder_block.forward(decoder_block2, skip1, sigmoid_depth_one_eight, decoder_block3);

        float sigmoid_depth_quarter[1][l2_out_height][l2_out_width];
        l2_depth_layer_quarter.forward(decoder_block3, sigmoid_depth_quarter);

        // float depth_quarter[l2_out_height][l2_out_width];
        // for (int j = 0; j < l2_out_height; j++) for (int k = 0; k < l2_out_width; k++)
        //     depth_quarter[j][k] = 1.0 / (inverse_depth_multiplier * sigmoid_depth_quarter[0][j][k] + inverse_depth_base);


        // 4th set
        const int l3_kernel_size = 5;
        const bool l3_plus_one = true;
        DecoderBlock<l2_out_channels, l2_out_height, l2_out_width, l3_kernel_size, apply_bn_relu, l3_plus_one> l3_decoder_block(param_path + "/decoder_block4");

        const int l3_out_channels = hyper_channels;
        const int l3_out_height = fe1_out_size(in_height);
        const int l3_out_width = fe1_out_size(in_width);
        depth_layer_3x3<l3_out_channels, l3_out_height, l3_out_width> l3_depth_layer_half(param_path + "/depth_layer_half");

        float decoder_block4[l3_out_channels][l3_out_height][l3_out_width];
        l3_decoder_block.forward(decoder_block3, skip0, sigmoid_depth_quarter, decoder_block4);

        float sigmoid_depth_half[1][l3_out_height][l3_out_width];
        l3_depth_layer_half.forward(decoder_block4, sigmoid_depth_half);

        // float depth_half[l3_out_height][l3_out_width];
        // for (int j = 0; j < l3_out_height; j++) for (int k = 0; k < l3_out_width; k++)
        //     depth_half[j][k] = 1.0 / (inverse_depth_multiplier * sigmoid_depth_half[0][j][k] + inverse_depth_base);


        // 5th set
        const int l4_in_height = l3_out_height * 2;
        const int l4_in_width = l3_out_width * 2;
        float scaled_depth[1][l4_in_height][l4_in_width];
        interpolate<1, l3_out_height, l3_out_width, l4_in_height, l4_in_width>(sigmoid_depth_half, scaled_depth, "bilinear");

        float scaled_decoder[l3_out_channels][l4_in_height][l4_in_width];
        interpolate<l3_out_channels, l3_out_height, l3_out_width, l4_in_height, l4_in_width>(decoder_block4, scaled_decoder, "bilinear");

        const int l4_in_channels = l3_out_channels + 4;
        float scaled_combined[l4_in_channels][l4_in_height][l4_in_width];
        for (int i = 0; i < l3_out_channels; i++) for (int j = 0; j < l4_in_height; j++) for (int k = 0; k < l4_in_width; k++)
            scaled_combined[i][j][k] = scaled_decoder[i][j][k];
        for (int j = 0; j < l4_in_height; j++) for (int k = 0; k < l4_in_width; k++)
            scaled_combined[l3_out_channels][j][k] = scaled_depth[0][j][k];
        for (int i = 0; i < 3; i++) for (int j = 0; j < l4_in_height; j++) for (int k = 0; k < l4_in_width; k++)
            scaled_combined[i+l3_out_channels+1][j][k] = image[i][j][k];

        const int l4_kernel_size = 5;
        const int l4_stride = 1;

        const int l4_out_channels = hyper_channels;
        const int l4_out_height = in_height;
        const int l4_out_width = in_width;
        conv_layer<l4_in_channels, l4_in_height, l4_in_width, l4_out_channels, l4_out_height, l4_out_width, l4_kernel_size, l4_stride, apply_bn_relu> l4_refine0(param_path + "/refine.0");
        conv_layer<l4_out_channels, l4_out_height, l4_out_width, l4_out_channels, l4_out_height, l4_out_width, l4_kernel_size, l4_stride, apply_bn_relu> l4_refine1(param_path + "/refine.1");
        depth_layer_3x3<l4_out_channels, l4_out_height, l4_out_width> l4_depth_layer_full(param_path + "/depth_layer_full");

        float refined0[l4_out_channels][l4_out_height][l4_out_width];
        l4_refine0.forward(scaled_combined, refined0);

        float refined1[l4_out_channels][l4_out_height][l4_out_width];
        l4_refine1.forward(refined0, refined1);

        float sigmoid_depth_full[1][l4_out_height][l4_out_width];
        l4_depth_layer_full.forward(refined1, sigmoid_depth_full);

        // float depth_full[l4_out_height][l4_out_width];
        for (int j = 0; j < l4_out_height; j++) for (int k = 0; k < l4_out_width; k++)
            depth_full[j][k] = 1.0 / (inverse_depth_multiplier * sigmoid_depth_full[0][j][k] + inverse_depth_base);

    }

private:
    string param_path;
};


template <int in_height, int in_width>
class LSTMFusion{
public:
    LSTMFusion(const string param_path) : param_path(param_path) {}

    void forward(const float image[3][in_height][in_width],
                 const float skip0[hyper_channels][fe1_out_size(in_height)][fe1_out_size(in_width)],
                 const float skip1[hyper_channels * 2][fe2_out_size(in_height)][fe2_out_size(in_width)],
                 const float skip2[hyper_channels * 4][fe3_out_size(in_height)][fe3_out_size(in_width)],
                 const float skip3[hyper_channels * 8][fe4_out_size(in_height)][fe4_out_size(in_width)],
                 const float bottom[hyper_channels * 16][fe5_out_size(in_height)][fe5_out_size(in_width)],
                 float depth_full[in_height][in_width]) {

        const int in_channels = hyper_channels * 16;
        const int hid_channels = hyper_channels * 16;

    }

private:
    string param_path;
};
