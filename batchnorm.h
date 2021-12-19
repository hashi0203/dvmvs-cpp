#pragma once
#include "settings.h"

template <int in_channels, int in_height, int in_width>
class BatchNorm2d{
public:
    BatchNorm2d() {
        // private 変数の初期値設定
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
    float running_mean[in_channels];
    float running_var[in_channels];
    float weight[in_channels];
    float bias[in_channels];
};
