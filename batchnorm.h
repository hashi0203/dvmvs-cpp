#pragma once
#include "settings.h"

void BatchNorm2d(float* x,
                const float *params, unordered_map<string, int> mp, const string param_path,
                const int channels, const int height, const int width) {

    if (mp.find(param_path + ".running_mean") == mp.end())
        cout << param_path + ".running_mean" << "\n";
    if (mp.find(param_path + ".running_var") == mp.end())
        cout << param_path + ".running_var" << "\n";
    if (mp.find(param_path + ".weight") == mp.end())
        cout << param_path + ".weight" << "\n";
    if (mp.find(param_path + ".bias") == mp.end())
        cout << param_path + ".bias" << "\n";

    const float* running_mean = params + mp[param_path + ".running_mean"];
    const float* running_var = params + mp[param_path + ".running_var"];
    const float* weight = params + mp[param_path + ".weight"];
    const float* bias = params + mp[param_path + ".bias"];

    // https://www.anarchive-beta.com/entry/2020/08/16/180000
    for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
        const int idx = (i * height + j) * width + k;
        const float xc = x[idx] - running_mean[i];
        const float xn = xc / sqrt(running_var[i] + 1e-5);
        x[idx] = weight[i] * xn + bias[i];
    }
}

// template <int channels, int height, int width>
// class BatchNorm2d{
// public:
//     float *running_mean = new float[channels];
//     float *running_var = new float[channels];
//     float *weight = new float[channels];
//     float *bias = new float[channels];

//     BatchNorm2d(const string param_path) : param_path(param_path) {
//         // load parameters
//         ifstream ifs;
//         ifs = open_file(param_path + ".running_mean");
//         ifs.read((char*) running_mean, sizeof(float) * channels);

//         ifs = open_file(param_path + ".running_var");
//         ifs.read((char*) running_var, sizeof(float) * channels);

//         ifs = open_file(param_path + ".weight");
//         ifs.read((char*) weight, sizeof(float) * channels);

//         ifs = open_file(param_path + ".bias");
//         ifs.read((char*) bias, sizeof(float) * channels);
//     }

//     void forward(const float x[channels][height][width], float y[channels][height][width]) {
//         // https://www.anarchive-beta.com/entry/2020/08/16/180000
//         for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
//             const float xc = x[i][j][k] - running_mean[i];
//             const float xn = xc / sqrt(running_var[i] + 1e-5);
//             y[i][j][k] = weight[i] * xn + bias[i];
//         }
//         close();
//     }

//     void close() {
//         delete[] running_mean;
//         delete[] running_var;
//         delete[] weight;
//         delete[] bias;
//     }

// private:
//     string param_path;
// };
