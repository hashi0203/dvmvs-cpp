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

    const float mm = (mshift > 0) ? 1 / (float) (1 << mshift) : 1 << (-mshift);
    const float vv = (vshift > 0) ? 1 / (float) (1 << vshift) : 1 << (-vshift);

    if (xshift - mshift < 0) print1("(xshift - mshift) is negative");
    if (xshift + wshift - bshift < 0) print1("(xshift + wshift - bshift) is negative");
    if (xshift + wshift - yshift < 0) print1("(xshift + wshift - yshift) is negative");

    // https://www.anarchive-beta.com/entry/2020/08/16/180000
    for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
        const int idx = (i * height + j) * width + k;
        const qmint rv = (vshift > 0) ? running_var[i] + voffset : (((qmint) running_var[i]) << (-vshift)) + voffset;
        // const float rv = (vshift > 0) ? (running_var[i] + voffset) * vv : (running_var[i] * vv) + voffset;

        // const float xc = x[idx] - (running_mean[i] * mm);
        // const float xn = xc / sqrt(rv + 1e-5);
        // const float rm = running_mean[i] * mm * weight[i];

        const qmint rm = (xshift > mshift) ? ((qmint) running_mean[i]) << (xshift - mshift) : running_mean[i] >> (mshift - xshift);
        const qmint xn = x[idx] * rv - rm;
        const qmint xnw = (vshift > 0) ? (xn * weight[i]) >> vshift : xn * weight[i];
        const qmint b = ((qmint) bias[i]) << (xshift + wshift - bshift);
        x[idx] = (xnw + b) >> (xshift + wshift - yshift);
        // const float yy = ((xnw - rm) / (float) ((1 << wshift) * (1 << xshift))) + (bias[i] / (float) (1 << bshift));
        // x[idx] = yy * (1 << yshift);

        // const float xn = (x[idx] / (float) (1 << xshift)) * rv - (running_mean[i] * mm);
        // x[idx] = (((weight[i] * xn) / (float) (1 << wshift)) + (bias[i] / (float) (1 << bshift))) * (1 << yshift);

        // const float xn = (x[idx] >> xshift) * rv - (running_mean[i] << (-mshift));
        // x[idx] = (((weight[i] * xn) >> wshift) + (bias[i] >> bshift)) << yshift;

        // const qaint rm = (xshift > mshift) ? running_mean[i] << (xshift-mshift) : running_mean[i] >> (mshift-xshift);
        // const qaint xn = x[idx] * rv - rm;
        // const qaint xnw = (vshift > 0) ? (xn * weight[i]) >> vshift : xn * weight[i];
        // x[idx] = (xnw + (bias[i] << (xshift + wshift - bshift))) >> (xshift + wshift - yshift);

        // if xshift - mshift can be negative
        // const qaint rm = (xshift > mshift) ? (running_mean[i] * weight[i]) << (xshift-mshift) : (running_mean[i] * weight[i]) >> (mshift-xshift);
        // const qaint xn = x[idx] * rv;
        // const qaint xnw = (vshift > 0) ? (xn * weight[i]) >> vshift : xn * weight[i];
        // x[idx] = (xnw - rm + (bias[i] << (xshift + wshift - bshift))) >> (xshift + wshift - yshift);
    }
}
