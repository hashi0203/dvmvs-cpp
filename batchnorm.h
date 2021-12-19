#pragma once
#include "settings.h"

template <int in_channels, int in_height, int in_width>
class BatchNorm2d{
public:
    float running_mean[in_channels];
    float running_var[in_channels];
    float weight[in_channels];
    float bias[in_channels];

    BatchNorm2d(const string param_path) : param_path(param_path) {
        // load parameters
        ifstream ifs;
        ifs = open_file(param_path + ".running_mean");
        ifs.read((char*) running_mean, sizeof(float) * in_channels);

        ifs = open_file(param_path + ".running_var");
        ifs.read((char*) running_var, sizeof(float) * in_channels);

        ifs = open_file(param_path + ".weight");
        ifs.read((char*) weight, sizeof(float) * in_channels);

        ifs = open_file(param_path + ".bias");
        ifs.read((char*) bias, sizeof(float) * in_channels);
    }

    void forward(const float x[in_channels][in_height][in_width], float y[in_channels][in_height][in_width]) {
        // https://www.anarchive-beta.com/entry/2020/08/16/180000
        for (int i = 0; i < in_channels; i++) for (int j = 0; j < in_height; j++) for (int k = 0; k < in_width; k++) {
            const float xc = x[i][j][k] - running_mean[i];
            const float xn = xc / sqrt(running_var[i] + 10e-7);
            y[i][j][k] = weight[i] * xn + bias[i];
        }
    }

private:
    string param_path;
//     float running_mean[in_channels];
//     float running_var[in_channels];
//     float weight[in_channels];
//     float bias[in_channels];
};
