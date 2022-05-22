#pragma once

void Conv2d(const qaint* input,
            qaint* output,
            const string param_path,
            const int in_channels, const int in_height, const int in_width,
            const int out_channels, const int out_height, const int out_width,
            const int kernel_size, const int stride, const int padding, const int groups, const bool apply_scale) {

    // https://ichi.pro/conv-2-d-saigo-ni-fuxowa-do-pasu-de-nani-ga-okoru-ka-o-rikaisuru-30488625459528

    // to check order of layers
    // print1(param_path + ".weight");
    // if (apply_bias) print1(param_path + ".bias");

    const int wshift = w_shifts[w_cnt];
    const qwint* weight = weights + w_idx[w_cnt++];
    const int bshift = b_shifts[b_cnt];
    const qbint* bias = biases + b_idx[b_cnt++];
    const int sshift = apply_scale ? s_shifts[s_cnt] : 0;
    const qsint* scale = apply_scale ? scales + s_idx[s_cnt++] : nullptr;

    const int xshift = a_shifts[a_cnt];
    const int yshift = a_shifts[++a_cnt];
    const int mshift = max(bshift, xshift + wshift);

    print_neg_shift(param_path, "wshift", wshift);
    print_neg_shift(param_path, "bshift", bshift);
    print_neg_shift(param_path, "bshift + sshift - yshift", bshift + sshift - yshift);

    // if (wshift < 0) print2(param_path + ".weight", wshift);
    // if (bshift < 0) print2(param_path + ".bias", bshift);
    // if (mshift + sshift - yshift < 0) print1("(bshift + sshift - yshift) is negative");

    // const int wshift = 0;
    // const float* weight = params_f + start_idx[param_cnt++];
    // const int bshift = 0;
    // const float* bias = apply_bias ? params_f + start_idx[param_cnt++] : nullptr;

    // if (bshift - (xshift + wshift) < 0) print4("(bshift - (xshift + wshift)) is negative", bshift, xshift, wshift);

    const int ocpg = out_channels / groups;
    const int icpg = in_channels / groups;
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < ocpg; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    const int och = g * ocpg + oc;
                    qmint sum = 0;

                    for (int ic = 0; ic < icpg; ic++) {
                        const int ich = g * icpg + ic;

                        for (int kh = 0; kh <= 2*padding; kh++) {
                            for (int kw = 0; kw <= 2*padding; kw++) {
                                const int ih = oh * stride + kh - padding;
                                const int iw = ow * stride + kw - padding;

                                const int input_idx = (ich * in_height + ih) * in_width + iw;
                                const int weight_idx = ((och * icpg + ic) * kernel_size + kh) * kernel_size + kw;

                                sum += (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) ? 0 :
                                        input[input_idx] * (qmint) weight[weight_idx];
                            }
                        }
                    }

                    sum <<= mshift - (xshift + wshift);
                    sum += bias[och] << (mshift - bshift);
                    sum = apply_scale ? sum * scale[och] : sum;
                    const int output_idx = (och * out_height + oh) * out_width + ow;
                    output[output_idx] = sum >> (mshift + sshift - yshift);
                }
            }
        }
    }
}
