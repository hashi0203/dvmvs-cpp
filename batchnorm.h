#pragma once

void BatchNorm2d(qaint* x, const string param_path,
                 const int channels, const int height, const int width) {

    // to check order of layers
    // print1(param_path + ".running_mean");
    // print1(param_path + ".running_var");
    // print1(param_path + ".weight");
    // print1(param_path + ".bias");

    const int xshift = actshifts[act_cnt++];
    const int yshift = actshifts[act_cnt];

    const int mshift = shifts[param_cnt];
    const qwint* running_mean = params + start_idx[param_cnt++];
    const int vshift = shifts[param_cnt];
    const qwint* running_var = params + start_idx[param_cnt++];
    const int voffset = offsets[offset_cnt++];
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

    if (xshift - mshift < 0) print1("(xshift - mshift) is negative");
    if (xshift + wshift - bshift < 0) print1("(xshift + wshift - bshift) is negative");
    if (xshift + wshift - yshift < 0) print1("(xshift + wshift - yshift) is negative");

    // https://www.anarchive-beta.com/entry/2020/08/16/180000
    for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
        const int idx = (i * height + j) * width + k;

        const qmint rv = (vshift > 0) ? running_var[i] + voffset : (((qmint) running_var[i]) << (-vshift)) + voffset;
        const qmint xn = x[idx] * rv;
        const qmint xnw = (vshift > 0) ? (xn * weight[i]) >> vshift : xn * weight[i];

        const qmint rm = (xshift > mshift) ? (running_mean[i] * (qmint) weight[i]) << (xshift - mshift) : (running_mean[i] * (qmint) weight[i]) >> (mshift - xshift);
        const qmint b = ((qmint) bias[i]) << (xshift + wshift - bshift);
        x[idx] = (xnw - rm + b) >> (xshift + wshift - yshift);
    }
}
