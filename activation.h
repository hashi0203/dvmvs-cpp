#pragma once
#include "settings.h"

template <int in_channels, int in_height, int in_width>
class ReLU{
public:
    void forward(const float x[in_channels][in_height][in_width], float y[in_channels][in_height][in_width]) {
        for (int i = 0; i < in_channels; i++) for (int j = 0; j < in_height; j++) for (int k = 0; k < in_width; k++)
            y[i][j][k] = (x[i][j][k] < 0) ? 0 : x[i][j][k];
    }
};
