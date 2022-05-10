#pragma once

void BatchNorm2d(float* x, const string param_path,
                 const int channels, const int height, const int width) {

    // to check order of layers
    // print1(param_path + ".running_mean");
    // print1(param_path + ".running_var");
    // print1(param_path + ".weight");
    // print1(param_path + ".bias");

    const int mshift = shifts[param_cnt];
    const qwint* running_mean = params + start_idx[param_cnt++];
    const int vshift = shifts[param_cnt];
    const qwint* running_var = params + start_idx[param_cnt++];
    const int wshift = shifts[param_cnt];
    const qwint* weight = params + start_idx[param_cnt++];
    const int bshift = shifts[param_cnt];
    const qwint* bias = params + start_idx[param_cnt++];

    if (mshift < 0) print2(param_path + ".running_mean", mshift);
    if (vshift < 0) print2(param_path + ".running_var", vshift);
    if (wshift < 0) print2(param_path + ".weight", wshift);
    if (bshift < 0) print2(param_path + ".bias", bshift);

    // const int mshift = 0;
    // const float* running_mean = params_f + start_idx[param_cnt++];
    // const int vshift = 0;
    // const float* running_var = params_f + start_idx[param_cnt++];
    // const int wshift = 0;
    // const float* weight = params_f + start_idx[param_cnt++];
    // const int bshift = 0;
    // const float* bias = params_f + start_idx[param_cnt++];

    const float mm = (mshift > 0) ? 1 / (float) (1 << mshift) : 1 << (-mshift);
    const float vv = (vshift > 0) ? 1 / (float) (1 << vshift) : 1 << (-vshift);

    // https://www.anarchive-beta.com/entry/2020/08/16/180000
    for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
        const int idx = (i * height + j) * width + k;
        const float xc = x[idx] - (running_mean[i] * mm);
        const float xn = xc / sqrt((running_var[i] * vv) + 1e-5);
        x[idx] = ((weight[i] * xn) / (float) (1 << wshift)) + (bias[i] / (float) (1 << bshift));
    }
}
