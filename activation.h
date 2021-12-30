#pragma once
#include "settings.h"

template <int channels, int height, int width>
class ReLU{
public:
    void forward(const float x[channels][height][width], float y[channels][height][width]) {
        for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            y[i][j][k] = max(0.0f, x[i][j][k]);
    }

    // void forward(float*** x, float y[channels][height][width]) {
    //     for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
    //         y[i][j][k] = (x[i][j][k] < 0) ? 0 : x[i][j][k];
    // }

    // void forward(const float x[channels][height][width], float*** y) {
    //     for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
    //         y[i][j][k] = (x[i][j][k] < 0) ? 0 : x[i][j][k];
    // }

    // void forward(float*** x, float*** y) {
    //     for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
    //         y[i][j][k] = (x[i][j][k] < 0) ? 0 : x[i][j][k];
    // }
};


template <int channels, int height, int width>
class celu{
public:
    void forward(const float x[channels][height][width], float y[channels][height][width]) {
        for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            y[i][j][k] = max(0.0f, x[i][j][k]) + min(0, exp(x[i][j][k]) - 1);
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
