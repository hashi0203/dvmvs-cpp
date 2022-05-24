#pragma once

void conv_layer(const qaint* x, qaint* y, const string param_path,
                const int in_channels, const int in_height, const int in_width,
                const int out_channels, const int out_height, const int out_width,
                const int kernel_size, const int stride) {

    const int padding = (kernel_size - 1) / 2;
    constexpr int groups = 1;
    constexpr bool apply_scale = true;
    // save_layer<qaint>("./results-qt/", "conv_x", "00009", x, in_channels * in_height * in_width, cin_shifts[conv_cnt]);
    Conv2d(x, y, param_path + ".0", in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, padding, groups, apply_scale);
    // print1(conv_cnt);
    // print1(bn_cnt);
    // save_layer<qaint>("./results-qt/", "conv_y", "00009", y, out_channels * out_height * out_width, cout_shifts[conv_cnt-1]);
    ReLU(y, out_channels, out_height, out_width);
}


// void depth_layer_3x3(const float* x, float* y, const string param_path,
//                      const int in_channels, const int height, const int width) {

//     constexpr int out_channels = 1;
//     constexpr int kernel_size = 3;
//     constexpr int stride = 1;
//     constexpr int padding = (kernel_size - 1) / 2;
//     constexpr int groups = 1;
//     constexpr bool apply_bias = true;
//     Conv2d(x, y, param_path + ".0", in_channels, height, width, out_channels, height, width, kernel_size, stride, padding, groups, apply_bias);
//     Sigmoid(y, out_channels, height, width);
// }
