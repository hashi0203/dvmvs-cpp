#pragma once
#include "settings.h"

template <int channels, int height, int width>
class BatchNorm2d{
public:
    float *running_mean = new float[channels];
    float *running_var = new float[channels];
    float *weight = new float[channels];
    float *bias = new float[channels];

    BatchNorm2d(const string param_path) : param_path(param_path) {
        // load parameters
        ifstream ifs;
        ifs = open_file(param_path + ".running_mean");
        ifs.read((char*) running_mean, sizeof(float) * channels);

        ifs = open_file(param_path + ".running_var");
        ifs.read((char*) running_var, sizeof(float) * channels);

        ifs = open_file(param_path + ".weight");
        ifs.read((char*) weight, sizeof(float) * channels);

        ifs = open_file(param_path + ".bias");
        ifs.read((char*) bias, sizeof(float) * channels);
    }

    void forward(const float x[channels][height][width], float y[channels][height][width]) {
        // https://www.anarchive-beta.com/entry/2020/08/16/180000
        for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
            const float xc = x[i][j][k] - running_mean[i];
            const float xn = xc / sqrt(running_var[i] + 1e-7);
            y[i][j][k] = weight[i] * xn + bias[i];
        }
        close();
    }

    void close() {
        delete[] running_mean;
        delete[] running_var;
        delete[] weight;
        delete[] bias;
    }

private:
    string param_path;
};
