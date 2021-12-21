#include "functional.h"

void grid_sample(const float image[fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)],
                 const float warping[warp_grid_height][warp_grid_width][2],
                 float warped_image[fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)]) {

    const int channels = fe1_out_channels;
    const int height = fe1_out_size(test_image_height);
    const int width = fe1_out_size(test_image_width);

    // float pad_image[fe1_out_channels][fe1_out_size(test_image_height)+2][fe1_out_size(test_image_width)+2];
    // pad_input<fe1_out_channels, fe1_out_size(test_image_height), fe1_out_size(test_image_width), 1>(image, pad_image);

    for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
        const float y = (warping[j][k][0] + 1) * height / 2.0;
        const float x = (warping[j][k][1] + 1) * width / 2.0;
        if (y < 0 || height-1 < y || x < 0 || width-1 < x) {
            for (int i = 0; i < channels; i++) warped_image[i][j][k] = 0;
        } else {
            const int y_int = min((int) y, height-2);
            const int x_int = min((int) x, width-2);
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
