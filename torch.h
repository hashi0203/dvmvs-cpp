#pragma once
#include "settings.h"

// inline const int conv_out_size(int size, int kernel_size, int stride, int padding) {
//     return (size + 2 * padding - kernel_size) / stride + 1;
// }

#define conv_out_size(size, kernel_size, stride, padding) ((size) + 2 * (padding) - (kernel_size)) / (stride) + 1

ifstream open_file(string path) {
    ifstream ifs(path);
    if (!ifs.is_open()) {
        cout << "File: " << path << " does not exist.\n";
        exit(1);
    }
    return ifs;
}

template<int channels, int in_height, int in_width, int out_height, int out_width>
void interpolate(float x[channels][in_height][in_width], float y[channels][out_height][out_width]) {
    const float fj = (float) in_height / out_height;
    const float fk = (float) in_width / out_width;
    for (int i = 0; i < channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++) {
        const int jj = round(j * fj);
        const int kk = round(k * fk);
        y[i][j][k] = x[i][jj][kk];
    }
}

// template<int channels, int in_height, int in_width, int padding>
// void pad_input(float input[channels][in_height][in_width], float output[channels][in_height+2*padding][in_width+2*padding]);

// template<int out_channels, int in_channels, int kernel_size>
// void kaiming_uniform_(float a, float weight[out_channels][in_channels][kernel_size]);

template<int channels, int in_height, int in_width, int padding>
void pad_input(float input[channels][in_height][in_width], float output[channels][in_height+2*padding][in_width+2*padding]) {
    for (int i = 0; i < channels; i++) for (int j = 0; j < in_height+2*padding; j++) for (int k = 0; k < in_width+2*padding; k++)
        output[i][j][k] = 0;
    for (int i = 0; i < channels; i++) for (int j = 0; j < in_height; j++) for (int k = 0; k < in_width; k++)
        output[i][j+padding][k+padding] = input[i][j][k];
}

float calculate_gain(float a) {
    return sqrt(2.0 / (1 + pow(a, 2)));
}

template<int out_channels, int in_channels, int kernel_size>
void kaiming_uniform_(float a, float weight[out_channels][in_channels][kernel_size][kernel_size]) {
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
