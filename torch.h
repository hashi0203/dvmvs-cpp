#pragma once
#include "settings.h"

ifstream open_file(const string path);

// template<int channels, int in_height, int in_width, int padding>
// void pad_input(float input[channels][in_height][in_width], float output[channels][in_height+2*padding][in_width+2*padding]);

// template<int out_channels, int in_channels, int kernel_size>
// void kaiming_uniform_(float a, float weight[out_channels][in_channels][kernel_size]);

template<int channels, int in_height, int in_width, int padding>
void pad_input(const float input[channels][in_height][in_width], float output[channels][in_height+2*padding][in_width+2*padding]) {
    for (int i = 0; i < channels; i++) for (int j = 0; j < in_height+2*padding; j++) for (int k = 0; k < in_width+2*padding; k++)
        output[i][j][k] = 0;
    for (int i = 0; i < channels; i++) for (int j = 0; j < in_height; j++) for (int k = 0; k < in_width; k++)
        output[i][j+padding][k+padding] = input[i][j][k];
}


void layer_norm(float* x, const int channels, const int height, const int width);

// template<int channels, int height, int width>
// void layer_norm(const float input[channels][height][width], float output[channels][height][width]) {
//     const float eps = 1e-5;
//     const int n1 = height * width;
//     for (int i = 0; i < channels; i++) {
//         float e = 0;
//         float v = 0;
//         for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
//             e += input[i][j][k];
//             v += input[i][j][k] * input[i][j][k];
//         }
//         e /= n1;
//         v /= n1;
//         v -= e * e;
//         for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
//             output[i][j][k] = (input[i][j][k] - e) / sqrt(v + eps);
//     }
// }


float calculate_gain(const float a);


template<int out_channels, int in_channels, int kernel_size>
void kaiming_uniform_(const float a, float weight[out_channels][in_channels][kernel_size][kernel_size]) {
    const float fan_in = in_channels * kernel_size * kernel_size; // 4 次元以下ならこれで OK なはず
    const float gain = calculate_gain(a);
    const float std = gain / sqrt(fan_in);
    const float bound = sqrt(3.0) * std; // Calculate uniform bounds from standard deviation
    random_device seed_gen;
    mt19937 engine(seed_gen());
    uniform_real_distribution<> uniform_dist(-bound, bound);
    for (int i = 0; i < out_channels; i++) for (int j = 0; j < in_channels; j++) {
        for (int k = 0; k < kernel_size; k++) for (int l = 0; l < kernel_size; l++)
            weight[i][j][k][l] = uniform_dist(engine);
    }
}
