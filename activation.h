#pragma once
#include "settings.h"

void ReLU(float* x, const int channels, const int height, const int width) {

    for (int idx = 0; idx < channels * height * width; idx++)
        x[idx] = max(0.0f, x[idx]);

}


template <int channels, int height, int width>
class celu{
public:
    void forward(const float x[channels][height][width], float y[channels][height][width]) {
        for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            y[i][j][k] = max(0.0f, x[i][j][k]) + min(0.0f, exp(x[i][j][k]) - 1);
    }
};


template <int channels, int height, int width>
class Sigmoid{
public:
    void forward(const float x[channels][height][width], float y[channels][height][width]) {
        for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            y[i][j][k] = 1.0 / (1.0 + exp(-x[i][j][k]));
    }
};
