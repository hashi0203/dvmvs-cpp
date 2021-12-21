#pragma once
#include "config.h"

template<int channels, int in_height, int in_width, int out_height, int out_width>
void interpolate(const float input[channels][in_height][in_width], float output[channels][out_height][out_width]) {
    const float fj = (float) in_height / out_height;
    const float fk = (float) in_width / out_width;
    for (int i = 0; i < channels; i++) for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++) {
        const int jj = round(j * fj);
        const int kk = round(k * fk);
        output[i][j][k] = input[i][jj][kk];
    }
}

void grid_sample(const float image[fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)],
                 const float warping[warp_grid_height][warp_grid_width][2],
                 float warped_image[fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)]);
