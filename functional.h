#pragma once
#include "config.h"

template<int channels, int in_height, int in_width, int out_height, int out_width>
void interpolate(const float input[channels][in_height][in_width], float output[channels][out_height][out_width], const string mode = "nearest") {
    const float fy = (float) in_height / out_height;
    const float fx = (float) in_width / out_width;
    if (mode == "nearest") {
        for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++) {
            const int y = (out_height == in_height * 2) ? j >> 1 : round(j * fy);
            const int x = (out_width == in_width * 2) ? k >> 1 : round(k * fx);
            for (int i = 0; i < channels; i++)
                output[i][j][k] = input[i][y][x];
        }
    } else if (mode == "bilinear") {
        float padded_input[channels][in_height+1][in_width+1];
        for (int i = 0; i < channels; i++) {
            for (int j = 0; j < in_height; j++) {
                for (int k = 0; k < in_width; k++)
                    padded_input[i][j][k] = input[i][j][k];
                padded_input[i][j][in_width] = input[i][j][in_width-1];
            }
            for (int k = 0; k < in_width; k++)
                padded_input[i][in_height][k] = input[i][in_height-1][k];
            padded_input[i][in_height][in_width] = input[i][in_height-1][in_width-1];
        }

        for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++) {
            const float y = j * fy;
            const float x = k * fx;
            const int y_int = y;
            const int x_int = x;
            const int ys[2] = {y_int, y_int + 1};
            const int xs[2] = {x_int, x_int + 1};
            const float dys[2] = {y - ys[0], ys[1] - y};
            const float dxs[2] = {x - xs[0], xs[1] - x};
            for (int i = 0; i < channels; i++) {
                output[i][j][k] = 0;
                for (int yi = 0; yi < 2; yi++) for (int xi = 0; xi < 2; xi++)
                    output[i][j][k] += dys[1-yi] * dxs[1-xi] * padded_input[i][ys[yi]][xs[xi]];
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

    // float pad_image[channels][height+2][width+2];
    // pad_input<channels, height, width, 1>(image, pad_image);

    for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
        const float x = (warping[j][k][0] + 1) * (width - 1) / 2.0; // maybe vice versa
        const float y = (warping[j][k][1] + 1) * (height - 1) / 2.0;
        if (isnan(y) || y < 0 || height-1 < y || isnan(x) || x < 0 || width-1 < x) {
            for (int i = 0; i < channels; i++) warped_image[i][j][k] = 0;
        } else {
            const int y_int = min(y, (float) height-2);
            const int x_int = min(x, (float) width-2);
            const int ys[2] = {y_int, y_int + 1};
            const int xs[2] = {x_int, x_int + 1};
            const float dys[2] = {y - ys[0], ys[1] - y};
            const float dxs[2] = {x - xs[0], xs[1] - x};
            for (int i = 0; i < channels; i++) {
                warped_image[i][j][k] = 0;
                for (int yi = 0; yi < 2; yi++) for (int xi = 0; xi < 2; xi++)
                    warped_image[i][j][k] += dys[1-yi] * dxs[1-xi] * image[i][ys[yi]][xs[xi]];
            }
        }
    }
}
