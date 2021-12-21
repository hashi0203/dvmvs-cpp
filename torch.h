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
