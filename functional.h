#pragma once
#include "config.h"

template<int channels, int in_height, int in_width, int out_height, int out_width>
void interpolate(const float input[channels][in_height][in_width], float output[channels][out_height][out_width], const string mode = "nearest") {
    const float fy = (float) in_height / out_height;
    const float fx = (float) in_width / out_width;
    if (mode == "nearest") {
        for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++) {
            const int y = j * fy + (fy - 1) / 2 + 0.5;
            const int x = k * fx + (fx - 1) / 2 + 0.5;
            for (int i = 0; i < channels; i++)
                output[i][j][k] = input[i][y][x];
        }
    } else if (mode == "bilinear") {
        for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++) {
            const float y = j * fy + (fy - 1) / 2;
            const float x = k * fx + (fx - 1) / 2;
            const int y_int = floor(y);
            const int x_int = floor(x);
            const int ys[2] = {y_int, y_int + 1};
            const int xs[2] = {x_int, x_int + 1};
            const float dys[2] = {y - ys[0], ys[1] - y};
            const float dxs[2] = {x - xs[0], xs[1] - x};
            for (int i = 0; i < channels; i++) {
                output[i][j][k] = 0;
                for (int yi = 0; yi < 2; yi++) for (int xi = 0; xi < 2; xi++) {
                    const float val = input[i][min(in_height-1, max(0, ys[yi]))][min(in_width-1, max(0, xs[xi]))];
                    output[i][j][k] += dys[1-yi] * dxs[1-xi] * val;
                }
            }
        }
    } else {
        cout << "The 'mode' option in interpolation should be 'nearest' or 'bilinear,' but it is " << mode << "\n";
        exit(1);
    }
}


template <int channels, int height, int width>
void grid_sample(const float image[channels][height][width],
                 const float warping[height][width][2],
                 float warped_image[channels][height][width]) {

    for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
        const float x = (warping[j][k][0] + 1) * (width - 1) / 2.0;
        const float y = (warping[j][k][1] + 1) * (height - 1) / 2.0;
        const int y_int = floor(y);
        const int x_int = floor(x);
        const int ys[2] = {y_int, y_int + 1};
        const int xs[2] = {x_int, x_int + 1};
        const float dys[2] = {y - ys[0], ys[1] - y};
        const float dxs[2] = {x - xs[0], xs[1] - x};
        for (int i = 0; i < channels; i++) {
            warped_image[i][j][k] = 0;
            for (int yi = 0; yi < 2; yi++) for (int xi = 0; xi < 2; xi++) {
                const float val = (ys[yi] < 0 || height-1 < ys[yi] || xs[xi] < 0 || width-1 < xs[xi]) ? 0 : image[i][ys[yi]][xs[xi]];
                warped_image[i][j][k] += dys[1-yi] * dxs[1-xi] * val;
            }
        }
    }
}
