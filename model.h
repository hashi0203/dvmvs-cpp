#pragma once
#include "config.h"
#include "functional.h"
#include "layers.h"
#include "mnasnet.h"
#include "convlstm.h"

// template <int channels, int height, int width, int kernel_size, bool apply_bn_relu>
// class StandardLayer{
// public:
//     StandardLayer(const string param_path) : param_path(param_path) {}

//     void forward(const float x[channels][height][width], float y[channels][height][width]) {
//         const int stride = 1;

//         const bool l0_apply_bn_relu = true;
//         conv_layer<channels, height, width, channels, height, width, kernel_size, stride, l0_apply_bn_relu> l0_conv_layer(param_path + ".conv1");
//         conv_layer<channels, height, width, channels, height, width, kernel_size, stride, apply_bn_relu> l1_conv_layer(param_path + ".conv2");

//         float y0[channels][height][width];
//         l0_conv_layer.forward(x, y0);

//         // float y1[channels][height][width];
//         l1_conv_layer.forward(y0, y);

//         // for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
//         //     y[i][j][k] = y1[i][j][k];
//     }

// private:
//     string param_path;
// };


// template <int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int kernel_size>
// class DownconvolutionLayer{
// public:
//     DownconvolutionLayer(const string param_path) : param_path(param_path) {}

//     void forward(const float x[in_channels][in_height][in_width], float y[out_channels][out_height][out_width]) {
//         const int stride = 2;
//         const bool apply_bn_relu = true;

//         conv_layer<in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, apply_bn_relu> l0_down_conv(param_path + ".down_conv");

//         // float y0[out_channels][out_height][out_width];
//         l0_down_conv.forward(x, y);

//         // for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
//         //     y[i][j][k] = y0[i][j][k];
//     }

// private:
//     string param_path;
// };


// template <int in_channels, int in_height, int in_width, int out_channels, int kernel_size>
// class UpconvolutionLayer{
// public:
//     UpconvolutionLayer(const string param_path) : param_path(param_path) {}

//     void forward(const float x[in_channels][in_height][in_width], float y[out_channels][in_height * 2][in_width * 2]) {
//         const int out_height = in_height * 2;
//         const int out_width = in_width * 2;
//         float up_x[in_channels][out_height][out_width];
//         interpolate<in_channels, in_height, in_width, out_height, out_width>(x, up_x, "bilinear");

//         const int stride = 1;
//         const bool apply_bn_relu = true;

//         conv_layer<in_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, apply_bn_relu> l0_conv_layer(param_path + ".conv");

//         // float y0[out_channels][out_height][out_width];
//         l0_conv_layer.forward(up_x, y);

//         // for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
//         //     y[i][j][k] = y0[i][j][k];
//     }

// private:
//     string param_path;
// };


// template <int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int kernel_size>
// class EncoderBlock{
// public:
//     EncoderBlock(const string param_path) : param_path(param_path) {}

//     void forward(const float x[in_channels][in_height][in_width], float y[out_channels][out_height][out_width]) {

//         DownconvolutionLayer<in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size> l0_down_convolution(param_path + ".down_convolution");
//         float y0[out_channels][out_height][out_width];
//         // float ***y0 = new float**[out_channels];
//         // new_3d(y0, out_channels, out_height, out_width);
//         l0_down_convolution.forward(x, y0);

//         const bool apply_bn_relu = true;
//         StandardLayer<out_channels, out_height, out_width, kernel_size, apply_bn_relu> l1_standard_convolution(param_path + ".standard_convolution");
//         // float y1[out_channels][out_height][out_width];
//         l1_standard_convolution.forward(y0, y);
//         // delete_3d(y0, out_channels, out_height, out_width);

//         // for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
//         //     y[i][j][k] = y1[i][j][k];
//     }

// private:
//     string param_path;
// };


// template <int in_channels, int in_height, int in_width, int kernel_size, bool apply_bn_relu, bool plus_one>
// class DecoderBlock{
// public:
//     DecoderBlock(const string param_path) : param_path(param_path) {}

//     void forward(const float x[in_channels][in_height][in_width],
//                  const float skip[in_channels / 2][in_height * 2][in_width * 2],
//                  const float depth[1][in_height][in_width],
//                  float y[in_channels / 2][in_height * 2][in_width * 2]) {

//         const int out_height = in_height * 2;
//         const int out_width = in_width * 2;
//         const int out_channels = in_channels / 2;

//         UpconvolutionLayer<in_channels, in_height, in_width, out_channels, kernel_size> l0_up_convolution(param_path + ".up_convolution");
//         float y0[out_channels][out_height][out_width];
//         l0_up_convolution.forward(x, y0);

//         const int stride = 1;

//         // Aggregate skip and upsampled input
//         const int l1_in_channels = plus_one ? in_channels +  1 : in_channels;

//         float x1[l1_in_channels][out_height][out_width];
//         for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
//             x1[i][j][k] = y0[i][j][k];
//         for (int i = 0; i < out_channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
//             x1[i+out_channels][j][k] = skip[i][j][k];
//         if (plus_one) {
//             float up_depth[1][out_height][out_width];
//             interpolate<1, in_height, in_width, out_height, out_width>(depth, up_depth, "bilinear");
//             for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++)
//                 x1[in_channels][j][k] = up_depth[0][j][k];
//         }

//         const bool l1_apply_bn_relu = true;
//         conv_layer<l1_in_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, l1_apply_bn_relu> l1_conv_layer(param_path + ".convolution1");
//         float y1[out_channels][out_height][out_width];
//         l1_conv_layer.forward(x1, y1);

//         // Learn from aggregation
//         const bool l2_apply_bn_relu = apply_bn_relu;
//         conv_layer<out_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, l2_apply_bn_relu> l2_conv_layer(param_path + ".convolution2");
//         l2_conv_layer.forward(y1, y);
//     }

// private:
//     string param_path;
// };


void FeatureExtractor(const float x[3 * test_image_height * test_image_width],
                        float layer1[channels_1 * height_2 * width_2],
                        float layer2[channels_2 * height_4 * width_4],
                        float layer3[channels_3 * height_8 * width_8],
                        float layer4[channels_4 * height_16 * width_16],
                        float layer5[channels_5 * height_32 * width_32]) {

    constexpr int depths[8] = {32, channels_1, channels_2, channels_3, 80, channels_4, 192, channels_5};
    const bool apply_bias = false;

    // First layer: regular conv.
    float x0[3 * test_image_height * test_image_width];
    for (int idx = 0; idx < 3 * test_image_height * test_image_width; idx++)
        x0[idx] = x[idx];

    const int l0_kernel_size = 3;
    const int l0_stride = 2;
    const int l0_padding = 1;
    const int l0_groups = 1;
    const int l0_out_channels = depths[0];
    const int l0_out_height = conv_out_size(test_image_height, l0_kernel_size, l0_stride, l0_padding);
    const int l0_out_width = conv_out_size(test_image_width, l0_kernel_size, l0_stride, l0_padding);
    float y0[l0_out_channels * l0_out_height * l0_out_width];
    Conv2d(x0, y0, "layer1.0", 3, test_image_height, test_image_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size, l0_stride, l0_padding, l0_groups, apply_bias);

    const int l1_out_channels = depths[0];
    const int l1_out_height = l0_out_height;
    const int l1_out_width = l0_out_width;
    BatchNorm2d(y0, "layer1.1", l1_out_channels, l1_out_height, l1_out_width);

    const int l2_out_channels = depths[0];
    const int l2_out_height = l1_out_height;
    const int l2_out_width = l1_out_width;
    ReLU(y0, l2_out_channels, l2_out_height, l2_out_width);

    // Depthwise separable, no skip.
    const int l3_kernel_size = 3;
    const int l3_stride = 1;
    const int l3_padding = 1;
    const int l3_groups = depths[0];
    const int l3_out_channels = depths[0];
    const int l3_out_height = conv_out_size(l2_out_height, l3_kernel_size, l3_stride, l3_padding);
    const int l3_out_width = conv_out_size(l2_out_width, l3_kernel_size, l3_stride, l3_padding);
    float y3[l3_out_channels * l3_out_height * l3_out_width];
    Conv2d(y0, y3, "layer1.3", l2_out_channels, l2_out_height, l2_out_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size, l3_stride, l3_padding, l3_groups, apply_bias);

    const int l4_out_channels = depths[0];
    const int l4_out_height = l3_out_height;
    const int l4_out_width = l3_out_width;
    BatchNorm2d(y3, "layer1.4", l4_out_channels, l4_out_height, l4_out_width);

    const int l5_out_channels = depths[0];
    const int l5_out_height = l4_out_height;
    const int l5_out_width = l4_out_width;
    ReLU(y3, l5_out_channels, l5_out_height, l5_out_width);

    const int l6_kernel_size = 1;
    const int l6_stride = 1;
    const int l6_padding = 0;
    const int l6_groups = 1;
    const int l6_out_channels = depths[1];
    const int l6_out_height = conv_out_size(l5_out_height, l6_kernel_size, l6_stride, l6_padding);
    const int l6_out_width = conv_out_size(l5_out_width, l6_kernel_size, l6_stride, l6_padding);
    Conv2d(y3, layer1, "layer1.6", l5_out_channels, l5_out_height, l5_out_width, l6_out_channels, l6_out_height, l6_out_width, l6_kernel_size, l6_stride, l6_padding, l6_groups, apply_bias);

    const int l7_out_channels = depths[1];
    const int l7_out_height = l6_out_height;
    const int l7_out_width = l6_out_width;
    BatchNorm2d(layer1, "layer1.7", l7_out_channels, l7_out_height, l7_out_width);

    // MNASNet blocks: stacks of inverted residuals.
    const int l8_kernel_size = 3;
    const int l8_stride = 2;
    const int l8_expansion_factor = 3;
    const int l8_repeats = 3;
    const int l8_out_channels = depths[2];
    const int l8_out_height = stack_out_size(l7_out_height, l8_kernel_size, l8_stride);
    const int l8_out_width = stack_out_size(l7_out_width, l8_kernel_size, l8_stride);
    _stack(layer1, layer2, "layer2.0", l7_out_channels, l7_out_height, l7_out_width, l8_out_channels, l8_out_height, l8_out_width, l8_kernel_size, l8_stride, l8_expansion_factor, l8_repeats);

    const int l9_kernel_size = 5;
    const int l9_stride = 2;
    const int l9_expansion_factor = 3;
    const int l9_repeats = 3;
    const int l9_out_channels = depths[3];
    const int l9_out_height = stack_out_size(l8_out_height, l9_kernel_size, l9_stride);
    const int l9_out_width = stack_out_size(l8_out_width, l9_kernel_size, l9_stride);
    _stack(layer2, layer3, "layer3.0", l8_out_channels, l8_out_height, l8_out_width, l9_out_channels, l9_out_height, l9_out_width, l9_kernel_size, l9_stride, l9_expansion_factor, l9_repeats);

    const int l10_kernel_size = 5;
    const int l10_stride = 2;
    const int l10_expansion_factor = 6;
    const int l10_repeats = 3;
    const int l10_out_channels = depths[4];
    const int l10_out_height = stack_out_size(l9_out_height, l10_kernel_size, l10_stride);
    const int l10_out_width = stack_out_size(l9_out_width, l10_kernel_size, l10_stride);
    float y10[l10_out_channels * l10_out_height * l10_out_width];
    _stack(layer3, y10, "layer4.0", l9_out_channels, l9_out_height, l9_out_width, l10_out_channels, l10_out_height, l10_out_width, l10_kernel_size, l10_stride, l10_expansion_factor, l10_repeats);

    const int l11_kernel_size = 3;
    const int l11_stride = 1;
    const int l11_expansion_factor = 6;
    const int l11_repeats = 2;
    const int l11_out_channels = depths[5];
    const int l11_out_height = stack_out_size(l10_out_height, l11_kernel_size, l11_stride);
    const int l11_out_width = stack_out_size(l10_out_width, l11_kernel_size, l11_stride);
    _stack(y10, layer4, "layer4.1", l10_out_channels, l10_out_height, l10_out_width, l11_out_channels, l11_out_height, l11_out_width, l11_kernel_size, l11_stride, l11_expansion_factor, l11_repeats);

    const int l12_kernel_size = 5;
    const int l12_stride = 2;
    const int l12_expansion_factor = 6;
    const int l12_repeats = 4;
    const int l12_out_channels = depths[6];
    const int l12_out_height = stack_out_size(l11_out_height, l12_kernel_size, l12_stride);
    const int l12_out_width = stack_out_size(l11_out_width, l12_kernel_size, l12_stride);
    float y12[l12_out_channels * l12_out_height * l12_out_width];
    _stack(layer4, y12, "layer5.0", l11_out_channels, l11_out_height, l11_out_width, l12_out_channels, l12_out_height, l12_out_width, l12_kernel_size, l12_stride, l12_expansion_factor, l12_repeats);

    const int l13_kernel_size = 3;
    const int l13_stride = 1;
    const int l13_expansion_factor = 6;
    const int l13_repeats = 1;
    const int l13_out_channels = depths[7];
    const int l13_out_height = stack_out_size(l12_out_height, l13_kernel_size, l13_stride);
    const int l13_out_width = stack_out_size(l12_out_width, l13_kernel_size, l13_stride);
    _stack(y12, layer5, "layer5.1", l12_out_channels, l12_out_height, l12_out_width, l13_out_channels, l13_out_height, l13_out_width, l13_kernel_size, l13_stride, l13_expansion_factor, l13_repeats);

}


// class FeatureShrinker{
// // Module that adds a FPN from on top of a set of feature maps. This is based on
// // `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
// // The feature maps are currently supposed to be in increasing depth order.
// // The input to the model is expected to be an OrderedDict[Tensor], containing
// // the feature maps on top of which the FPN will be added.
// public:
//     FeatureShrinker(const string param_path) : param_path(param_path) {}

//     void forward(const float layer1[channels_1][height_2][width_2],
//                  const float layer2[channels_2][height_4][width_4],
//                  const float layer3[channels_3][height_8][width_8],
//                  const float layer4[channels_4][height_16][width_16],
//                  const float layer5[channels_5][height_32][width_32],
//                  float features_half[fpn_output_channels][height_2][width_2],
//                  float features_quarter[fpn_output_channels][height_4][width_4],
//                  float features_one_eight[fpn_output_channels][height_8][width_8],
//                  float features_one_sixteen[fpn_output_channels][height_16][width_16]) {

//         const int stride = 1;
//         const int groups = 1;
//         const bool apply_bias = true;

//         const int inner_kernel_size = 1;
//         const int inner_padding = 0;
//         const int layer_kernel_size = 3;
//         const int layer_padding = 1;

//         // layer5
//         Conv2d<channels_5, height_32, width_32, fpn_output_channels, height_32, width_32, inner_kernel_size, stride, inner_padding, groups, apply_bias> i5_conv(param_path + "/fpn.inner_blocks.4");
//         float inner5[fpn_output_channels][height_32][width_32];
//         i5_conv.forward(layer5, inner5);

//         // Conv2d<fpn_output_channels, height_32, width_32, fpn_output_channels, height_32, width_32, layer_kernel_size, stride, layer_padding, groups, apply_bias> l5_conv(param_path + "/fpn.layer_blocks.4");
//         // float features_smallest[fpn_output_channels][height_32][width_32];
//         // l5_conv.forward(inner5, features_smallest);


//         // layer4
//         Conv2d<channels_4, height_16, width_16, fpn_output_channels, height_16, width_16, inner_kernel_size, stride, inner_padding, groups, apply_bias> i4_conv(param_path + "/fpn.inner_blocks.3");
//         float inner4[fpn_output_channels][height_16][width_16];
//         i4_conv.forward(layer4, inner4);

//         float top_down4[fpn_output_channels][height_16][width_16];
//         interpolate<fpn_output_channels, height_32, width_32, height_16, width_16>(inner5, top_down4);
//         for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < height_16; j++) for (int k = 0; k < width_16; k++)
//             inner4[i][j][k] += top_down4[i][j][k];

//         Conv2d<fpn_output_channels, height_16, width_16, fpn_output_channels, height_16, width_16, layer_kernel_size, stride, layer_padding, groups, apply_bias> l4_conv(param_path + "/fpn.layer_blocks.3");
//         l4_conv.forward(inner4, features_one_sixteen);


//         // layer3
//         Conv2d<channels_3, height_8, width_8, fpn_output_channels, height_8, width_8, inner_kernel_size, stride, inner_padding, groups, apply_bias> i3_conv(param_path + "/fpn.inner_blocks.2");
//         float inner3[fpn_output_channels][height_8][width_8];
//         i3_conv.forward(layer3, inner3);

//         float top_down3[fpn_output_channels][height_8][width_8];
//         interpolate<fpn_output_channels, height_16, width_16, height_8, width_8>(inner4, top_down3);
//         for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < height_8; j++) for (int k = 0; k < width_8; k++)
//             inner3[i][j][k] += top_down3[i][j][k];

//         Conv2d<fpn_output_channels, height_8, width_8, fpn_output_channels, height_8, width_8, layer_kernel_size, stride, layer_padding, groups, apply_bias> l3_conv(param_path + "/fpn.layer_blocks.2");
//         l3_conv.forward(inner3, features_one_eight);


//         // layer2
//         Conv2d<channels_2, height_4, width_4, fpn_output_channels, height_4, width_4, inner_kernel_size, stride, inner_padding, groups, apply_bias> i2_conv(param_path + "/fpn.inner_blocks.1");
//         float inner2[fpn_output_channels][height_4][width_4];
//         i2_conv.forward(layer2, inner2);

//         float top_down2[fpn_output_channels][height_4][width_4];
//         interpolate<fpn_output_channels, height_8, width_8, height_4, width_4>(inner3, top_down2);
//         for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < height_4; j++) for (int k = 0; k < width_4; k++)
//             inner2[i][j][k] += top_down2[i][j][k];

//         Conv2d<fpn_output_channels, height_4, width_4, fpn_output_channels, height_4, width_4, layer_kernel_size, stride, layer_padding, groups, apply_bias> l2_conv(param_path + "/fpn.layer_blocks.1");
//         l2_conv.forward(inner2, features_quarter);


//         // layer1
//         Conv2d<channels_1, height_2, width_2, fpn_output_channels, height_2, width_2, inner_kernel_size, stride, inner_padding, groups, apply_bias> i1_conv(param_path + "/fpn.inner_blocks.0");
//         float inner1[fpn_output_channels][height_2][width_2];
//         i1_conv.forward(layer1, inner1);

//         float top_down1[fpn_output_channels][height_2][width_2];
//         interpolate<fpn_output_channels, height_4, width_4, height_2, width_2>(inner2, top_down1);
//         for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < height_2; j++) for (int k = 0; k < width_2; k++)
//             inner1[i][j][k] += top_down1[i][j][k];

//         Conv2d<fpn_output_channels, height_2, width_2, fpn_output_channels, height_2, width_2, layer_kernel_size, stride, layer_padding, groups, apply_bias> l1_conv(param_path + "/fpn.layer_blocks.0");
//         l1_conv.forward(inner1, features_half);
//     }

// private:
//     string param_path;
// };


// class CostVolumeEncoder{
// public:
//     CostVolumeEncoder(const string param_path) : param_path(param_path) {}

//     void forward(const float features_half[fpn_output_channels][height_2][width_2],
//                  const float features_quarter[fpn_output_channels][height_4][width_4],
//                  const float features_one_eight[fpn_output_channels][height_8][width_8],
//                  const float features_one_sixteen[fpn_output_channels][height_16][width_16],
//                  const float cost_volume[n_depth_levels][height_2][width_2],
//                  float skip0[hyper_channels][height_2][width_2],
//                  float skip1[hyper_channels * 2][height_4][width_4],
//                  float skip2[hyper_channels * 4][height_8][width_8],
//                  float skip3[hyper_channels * 8][height_16][width_16],
//                  float bottom[hyper_channels * 16][height_32][width_32]) {

//         const int stride = 1;
//         const bool apply_bn_relu = true;


//         // 1st set
//         const int l0_kernel_size = 5;
//         const int l0_in_channels = fpn_output_channels + n_depth_levels;

//         const int l0_mid_channels = hyper_channels;
//         const int l0_mid_height = height_2;
//         const int l0_mid_width = width_2;

//         const int l0_out_channels = hyper_channels * 2;
//         const int l0_out_height = height_4;
//         const int l0_out_width = width_4;

//         conv_layer<l0_in_channels, l0_mid_height, l0_mid_width, l0_mid_channels, l0_mid_height, l0_mid_width, l0_kernel_size, stride, apply_bn_relu> l0_aggregator(param_path + "/aggregator0");
//         EncoderBlock<l0_mid_channels, l0_mid_height, l0_mid_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size> l0_encoder_block(param_path + "/encoder_block0");

//         float l0_in[l0_in_channels][height_2][width_2];
//         for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < l0_mid_height; j++) for (int k = 0; k < l0_mid_width; k++)
//             l0_in[i][j][k] = features_half[i][j][k];
//         for (int i = 0; i < n_depth_levels; i++) for (int j = 0; j < l0_mid_height; j++) for (int k = 0; k < l0_mid_width; k++)
//             l0_in[i+fpn_output_channels][j][k] = cost_volume[i][j][k];
//         l0_aggregator.forward(l0_in, skip0);

//         float l0_out[l0_out_channels][l0_out_height][l0_out_width];
//         l0_encoder_block.forward(skip0, l0_out);


//         // 2nd set
//         const int l1_kernel_size = 3;
//         const int l1_in_channels = fpn_output_channels + l0_out_channels;

//         const int l1_mid_channels = hyper_channels * 2;
//         const int l1_mid_height = height_4;
//         const int l1_mid_width = width_4;

//         const int l1_out_channels = hyper_channels * 4;
//         const int l1_out_height = height_8;
//         const int l1_out_width = width_8;

//         conv_layer<l1_in_channels, l1_mid_height, l1_mid_width, l1_mid_channels, l1_mid_height, l1_mid_width, l1_kernel_size, stride, apply_bn_relu> l1_aggregator(param_path + "/aggregator1");
//         EncoderBlock<l1_mid_channels, l1_mid_height, l1_mid_width, l1_out_channels, l1_out_height, l1_out_width, l1_kernel_size> l1_encoder_block(param_path + "/encoder_block1");

//         float l1_in[l1_in_channels][height_4][width_4];
//         for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < l1_mid_height; j++) for (int k = 0; k < l1_mid_width; k++)
//             l1_in[i][j][k] = features_quarter[i][j][k];
//         for (int i = 0; i < l0_out_channels; i++) for (int j = 0; j < l1_mid_height; j++) for (int k = 0; k < l1_mid_width; k++)
//             l1_in[i+fpn_output_channels][j][k] = l0_out[i][j][k];
//         l1_aggregator.forward(l1_in, skip1);

//         float l1_out[l1_out_channels][l1_out_height][l1_out_width];
//         l1_encoder_block.forward(skip1, l1_out);


//         // 3rd set
//         const int l2_kernel_size = 3;
//         const int l2_in_channels = fpn_output_channels + l1_out_channels;

//         const int l2_mid_channels = hyper_channels * 4;
//         const int l2_mid_height = height_8;
//         const int l2_mid_width = width_8;

//         const int l2_out_channels = hyper_channels * 8;
//         const int l2_out_height = height_16;
//         const int l2_out_width = width_16;

//         conv_layer<l2_in_channels, l2_mid_height, l2_mid_width, l2_mid_channels, l2_mid_height, l2_mid_width, l2_kernel_size, stride, apply_bn_relu> l2_aggregator(param_path + "/aggregator2");
//         EncoderBlock<l2_mid_channels, l2_mid_height, l2_mid_width, l2_out_channels, l2_out_height, l2_out_width, l2_kernel_size> l2_encoder_block(param_path + "/encoder_block2");

//         float l2_in[l2_in_channels][height_8][width_8];
//         for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < l2_mid_height; j++) for (int k = 0; k < l2_mid_width; k++)
//             l2_in[i][j][k] = features_one_eight[i][j][k];
//         for (int i = 0; i < l1_out_channels; i++) for (int j = 0; j < l2_mid_height; j++) for (int k = 0; k < l2_mid_width; k++)
//             l2_in[i+fpn_output_channels][j][k] = l1_out[i][j][k];
//         l2_aggregator.forward(l2_in, skip2);

//         float l2_out[l2_out_channels][l2_out_height][l2_out_width];
//         l2_encoder_block.forward(skip2, l2_out);


//         // 4th set
//         const int l3_kernel_size = 3;
//         const int l3_in_channels = fpn_output_channels + l2_out_channels;

//         const int l3_mid_channels = hyper_channels * 8;
//         const int l3_mid_height = height_16;
//         const int l3_mid_width = width_16;

//         const int l3_out_channels = hyper_channels * 16;
//         const int l3_out_height = height_32;
//         const int l3_out_width = width_32;

//         conv_layer<l3_in_channels, l3_mid_height, l3_mid_width, l3_mid_channels, l3_mid_height, l3_mid_width, l3_kernel_size, stride, apply_bn_relu> l3_aggregator(param_path + "/aggregator3");
//         EncoderBlock<l3_mid_channels, l3_mid_height, l3_mid_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size> l3_encoder_block(param_path + "/encoder_block3");

//         float l3_in[l3_in_channels][height_16][width_16];
//         for (int i = 0; i < fpn_output_channels; i++) for (int j = 0; j < l3_mid_height; j++) for (int k = 0; k < l3_mid_width; k++)
//             l3_in[i][j][k] = features_one_sixteen[i][j][k];
//         for (int i = 0; i < l2_out_channels; i++) for (int j = 0; j < l3_mid_height; j++) for (int k = 0; k < l3_mid_width; k++)
//             l3_in[i+fpn_output_channels][j][k] = l2_out[i][j][k];
//         l3_aggregator.forward(l3_in, skip3);

//         // float l3_out[l3_out_channels][l3_out_height][l3_out_width];
//         l3_encoder_block.forward(skip3, bottom);

//         // for (int i = 0; i < l3_out_channels; i++) for (int j = 0; j < l3_out_height; j++) for (int k = 0; k < l3_out_width; k++)
//         //     bottom[i][j][k] = l3_out[i][j][k];

//     }

// private:
//     string param_path;
// };


// class CostVolumeDecoder{
// public:
//     CostVolumeDecoder(const string param_path) : param_path(param_path) {}

//     void forward(const float image[3 * test_image_height * test_image_width],
//                  const float skip0[hyper_channels][height_2][width_2],
//                  const float skip1[hyper_channels * 2][height_4][width_4],
//                  const float skip2[hyper_channels * 4][height_8][width_8],
//                  const float skip3[hyper_channels * 8][height_16][width_16],
//                  const float bottom[hyper_channels * 16][height_32][width_32],
//                  float depth_full[test_image_height][test_image_width]) {

//         const float inverse_depth_base = 1 / max_depth;
//         const float inverse_depth_multiplier = 1 / min_depth - 1 / max_depth;

//         const bool apply_bn_relu = true;

//         // 1st set
//         const int l0_in_channels = hyper_channels * 16;
//         const int l0_in_height = height_32;
//         const int l0_in_width = width_32;

//         const int l0_kernel_size = 3;
//         const bool l0_plus_one = false;
//         DecoderBlock<l0_in_channels, l0_in_height, l0_in_width, l0_kernel_size, apply_bn_relu, l0_plus_one> l0_decoder_block(param_path + "/decoder_block1");

//         const int l0_out_channels = hyper_channels * 8;
//         const int l0_out_height = height_16;
//         const int l0_out_width = width_16;
//         depth_layer_3x3<l0_out_channels, l0_out_height, l0_out_width> l0_depth_layer_one_sixteen(param_path + "/depth_layer_one_sixteen");

//         float null_depth[1][l0_in_height][l0_in_width];
//         float decoder_block1[l0_out_channels][l0_out_height][l0_out_width];
//         l0_decoder_block.forward(bottom, skip3, null_depth, decoder_block1);

//         float sigmoid_depth_one_sixteen[1][l0_out_height][l0_out_width];
//         l0_depth_layer_one_sixteen.forward(decoder_block1, sigmoid_depth_one_sixteen);

//         // float depth_one_sixteen[l0_out_height][l0_out_width];
//         // for (int j = 0; j < l0_out_height; j++) for (int k = 0; k < l0_out_width; k++)
//         //     depth_one_sixteen[j][k] = 1.0 / (inverse_depth_multiplier * sigmoid_depth_one_sixteen[0][j][k] + inverse_depth_base);


//         // 2nd set
//         const int l1_kernel_size = 3;
//         const bool l1_plus_one = true;
//         DecoderBlock<l0_out_channels, l0_out_height, l0_out_width, l1_kernel_size, apply_bn_relu, l1_plus_one> l1_decoder_block(param_path + "/decoder_block2");

//         const int l1_out_channels = hyper_channels * 4;
//         const int l1_out_height = height_8;
//         const int l1_out_width = width_8;
//         depth_layer_3x3<l1_out_channels, l1_out_height, l1_out_width> l1_depth_layer_one_eight(param_path + "/depth_layer_one_eight");

//         float decoder_block2[l1_out_channels][l1_out_height][l1_out_width];
//         l1_decoder_block.forward(decoder_block1, skip2, sigmoid_depth_one_sixteen, decoder_block2);

//         float sigmoid_depth_one_eight[1][l1_out_height][l1_out_width];
//         l1_depth_layer_one_eight.forward(decoder_block2, sigmoid_depth_one_eight);

//         // float depth_one_eight[l1_out_height][l1_out_width];
//         // for (int j = 0; j < l1_out_height; j++) for (int k = 0; k < l1_out_width; k++)
//         //     depth_one_eight[j][k] = 1.0 / (inverse_depth_multiplier * sigmoid_depth_one_eight[0][j][k] + inverse_depth_base);


//         // 3rd set
//         const int l2_kernel_size = 3;
//         const bool l2_plus_one = true;
//         DecoderBlock<l1_out_channels, l1_out_height, l1_out_width, l2_kernel_size, apply_bn_relu, l2_plus_one> l2_decoder_block(param_path + "/decoder_block3");

//         const int l2_out_channels = hyper_channels * 2;
//         const int l2_out_height = height_4;
//         const int l2_out_width = width_4;
//         depth_layer_3x3<l2_out_channels, l2_out_height, l2_out_width> l2_depth_layer_quarter(param_path + "/depth_layer_quarter");

//         float decoder_block3[l2_out_channels][l2_out_height][l2_out_width];
//         l2_decoder_block.forward(decoder_block2, skip1, sigmoid_depth_one_eight, decoder_block3);

//         float sigmoid_depth_quarter[1][l2_out_height][l2_out_width];
//         l2_depth_layer_quarter.forward(decoder_block3, sigmoid_depth_quarter);

//         // float depth_quarter[l2_out_height][l2_out_width];
//         // for (int j = 0; j < l2_out_height; j++) for (int k = 0; k < l2_out_width; k++)
//         //     depth_quarter[j][k] = 1.0 / (inverse_depth_multiplier * sigmoid_depth_quarter[0][j][k] + inverse_depth_base);


//         // 4th set
//         const int l3_kernel_size = 5;
//         const bool l3_plus_one = true;
//         DecoderBlock<l2_out_channels, l2_out_height, l2_out_width, l3_kernel_size, apply_bn_relu, l3_plus_one> l3_decoder_block(param_path + "/decoder_block4");

//         const int l3_out_channels = hyper_channels;
//         const int l3_out_height = height_2;
//         const int l3_out_width = width_2;
//         depth_layer_3x3<l3_out_channels, l3_out_height, l3_out_width> l3_depth_layer_half(param_path + "/depth_layer_half");

//         float decoder_block4[l3_out_channels][l3_out_height][l3_out_width];
//         l3_decoder_block.forward(decoder_block3, skip0, sigmoid_depth_quarter, decoder_block4);

//         float sigmoid_depth_half[1][l3_out_height][l3_out_width];
//         l3_depth_layer_half.forward(decoder_block4, sigmoid_depth_half);

//         // float depth_half[l3_out_height][l3_out_width];
//         // for (int j = 0; j < l3_out_height; j++) for (int k = 0; k < l3_out_width; k++)
//         //     depth_half[j][k] = 1.0 / (inverse_depth_multiplier * sigmoid_depth_half[0][j][k] + inverse_depth_base);


//         // 5th set
//         const int l4_in_height = l3_out_height * 2;
//         const int l4_in_width = l3_out_width * 2;
//         float scaled_depth[1][l4_in_height][l4_in_width];
//         interpolate<1, l3_out_height, l3_out_width, l4_in_height, l4_in_width>(sigmoid_depth_half, scaled_depth, "bilinear");

//         float scaled_decoder[l3_out_channels][l4_in_height][l4_in_width];
//         interpolate<l3_out_channels, l3_out_height, l3_out_width, l4_in_height, l4_in_width>(decoder_block4, scaled_decoder, "bilinear");

//         const int l4_in_channels = l3_out_channels + 4;
//         float scaled_combined[l4_in_channels][l4_in_height][l4_in_width];
//         for (int i = 0; i < l3_out_channels; i++) for (int j = 0; j < l4_in_height; j++) for (int k = 0; k < l4_in_width; k++)
//             scaled_combined[i][j][k] = scaled_decoder[i][j][k];
//         for (int j = 0; j < l4_in_height; j++) for (int k = 0; k < l4_in_width; k++)
//             scaled_combined[l3_out_channels][j][k] = scaled_depth[0][j][k];
//         for (int i = 0; i < 3; i++) for (int j = 0; j < l4_in_height; j++) for (int k = 0; k < l4_in_width; k++)
//             scaled_combined[i+l3_out_channels+1][j][k] = image[(i * test_image_height + j) * test_image_width + k];

//         const int l4_kernel_size = 5;
//         const int l4_stride = 1;

//         const int l4_out_channels = hyper_channels;
//         const int l4_out_height = test_image_height;
//         const int l4_out_width = test_image_width;
//         conv_layer<l4_in_channels, l4_in_height, l4_in_width, l4_out_channels, l4_out_height, l4_out_width, l4_kernel_size, l4_stride, apply_bn_relu> l4_refine0(param_path + "/refine.0");
//         conv_layer<l4_out_channels, l4_out_height, l4_out_width, l4_out_channels, l4_out_height, l4_out_width, l4_kernel_size, l4_stride, apply_bn_relu> l4_refine1(param_path + "/refine.1");
//         depth_layer_3x3<l4_out_channels, l4_out_height, l4_out_width> l4_depth_layer_full(param_path + "/depth_layer_full");

//         float refined0[l4_out_channels][l4_out_height][l4_out_width];
//         l4_refine0.forward(scaled_combined, refined0);

//         float refined1[l4_out_channels][l4_out_height][l4_out_width];
//         l4_refine1.forward(refined0, refined1);

//         float sigmoid_depth_full[1][l4_out_height][l4_out_width];
//         l4_depth_layer_full.forward(refined1, sigmoid_depth_full);

//         // float depth_full[l4_out_height][l4_out_width];
//         for (int j = 0; j < l4_out_height; j++) for (int k = 0; k < l4_out_width; k++)
//             depth_full[j][k] = 1.0 / (inverse_depth_multiplier * sigmoid_depth_full[0][j][k] + inverse_depth_base);

//     }

// private:
//     string param_path;
// };


// class LSTMFusion{
// public:
//     LSTMFusion(const string param_path) : param_path(param_path) {}

//     void forward(const float current_encoding[hyper_channels * 16][height_32][width_32],
//                  float hidden_state[hyper_channels * 16][height_32][width_32],
//                  float cell_state[hyper_channels * 16][height_32][width_32]) {

//         const int in_channels = hyper_channels * 16;
//         const int hid_channels = hyper_channels * 16;

//         const int kernel_size = 3;
//         MVSLayernormConvLSTMCell<in_channels, hid_channels, height_32, width_32, kernel_size> lstm_cell(param_path + "/lstm_cell");
//         lstm_cell.forward(current_encoding, hidden_state, cell_state);
//     }

// private:
//     string param_path;
// };
