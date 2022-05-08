#include "torch.h"

ifstream open_file(const string path) {
    ifstream ifs(path);
    if (!ifs.is_open()) {
        cerr << "Failed to open file: " << path << ".\n";
        exit(1);
    }
    return ifs;
}


void layer_norm(float* x, const int channels, const int height, const int width) {

    constexpr float eps = 1e-5;
    const int n1 = height * width;
    for (int i = 0; i < channels; i++) {
        float e = 0.f;
        float v = 0.f;
        for (int idx = 0; idx < height * width; idx++) {
            e += x[i * (height * width) + idx];
            v += x[i * (height * width) + idx] * x[i * (height * width) + idx];
        }
        e /= n1;
        v /= n1;
        v -= e * e;
        for (int idx = 0; idx < height * width; idx++)
            x[i * (height * width) + idx] = (x[i * (height * width) + idx] - e) / sqrt(v + eps);
    }
}


float calculate_gain(const float a) {
    return sqrt(2.0 / (1 + pow(a, 2)));
}