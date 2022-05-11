#pragma once

void ReLU(qaint* x, const int channels, const int height, const int width) {
    for (int idx = 0; idx < channels * height * width; idx++)
        x[idx] = max(0, x[idx]);
}


void celu(qaint* x, const int channels, const int height, const int width) {
    const int xshift = actshifts[act_cnt];
    for (int idx = 0; idx < channels * height * width; idx++) {
        const float xx = x[idx] / (float) (1 << xshift);
        x[idx] = max(0, x[idx]) + min(0, (int) ((exp(xx) - 1) * (1 << xshift)));
    }
}


void Sigmoid(qaint* x, const int channels, const int height, const int width) {
    const int xshift = actshifts[act_cnt];
    for (int idx = 0; idx < channels * height * width; idx++) {
        const float xx = x[idx] / (float) (1 << xshift);
        x[idx] = 1.0 / (1.0 + exp(-xx)) * (1 << xshift);
    }
}
