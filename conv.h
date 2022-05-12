#pragma once

constexpr int check[81] = {1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 201, 206, 211, 216, 221, 226, 231, 236, 241, 246, 251, 274, 279, 284, 289, 294, 299, 304, 309, 314, 319, 324, 329, 334, 339, 344, 349, 355, 360, 365, 372, 377, 382, 389, 394, 399, 406, 411, 416, 423, 428};


void Conv2d(const float* input,
            float* output,
            const string param_path,
            const int in_channels, const int in_height, const int in_width,
            const int out_channels, const int out_height, const int out_width,
            const int kernel_size, const int stride, const int padding, const int groups, const bool apply_bias) {

    // https://ichi.pro/conv-2-d-saigo-ni-fuxowa-do-pasu-de-nani-ga-okoru-ka-o-rikaisuru-30488625459528
    const float* weightC = params + start_idx[param_cnt++];
    const float* biasC = apply_bias ? params + start_idx[param_cnt++] : nullptr;
    // print1(param_path + ".weight");
    // if (apply_bias) print1(param_path + ".bias");

    const int ocpg = out_channels / groups;
    const int icpg = in_channels / groups;

    float* running_mean;
    float* running_var;
    float* weightB;
    float* biasB;

    bool bn = false;
    for (int i = 0; i < 81; i++) {
        if (param_cnt == check[i]) bn = true;
    }
    if (bn) {
        running_mean = params + start_idx[param_cnt++];
        running_var = params + start_idx[param_cnt++];
        weightB = params + start_idx[param_cnt++];
        biasB = params + start_idx[param_cnt++];
    }

    float* weight = new float[out_channels * icpg * kernel_size * kernel_size];
    float* bias = new float[out_channels];
    for (int och = 0; och < out_channels; och++) {

        float sum = (apply_bias) ? biasC[och] : 0.f;
        const float wrv = bn ? weightB[och] / sqrt(running_var[och] + 1e-5) : 1.0f;
        if (bn) {
            sum *= wrv;
            sum += biasB[och];
            sum -= running_mean[och] * wrv;
        }
        bias[och] = sum;

        for (int ic = 0; ic < icpg; ic++) {
            for (int kh = 0; kh <= 2*padding; kh++) {
                for (int kw = 0; kw <= 2*padding; kw++) {
                    const int weight_idx = ((och * icpg + ic) * kernel_size + kh) * kernel_size + kw;
                    weight[weight_idx] = weightC[weight_idx] * wrv;
                }
            }
        }
    }


    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < ocpg; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    const int och = g * ocpg + oc;
                    float sum = bias[och];

                    for (int ic = 0; ic < icpg; ic++) {
                        const int ich = g * icpg + ic;

                        for (int kh = 0; kh <= 2*padding; kh++) {
                            for (int kw = 0; kw <= 2*padding; kw++) {
                                const int ih = oh * stride + kh - padding;
                                const int iw = ow * stride + kw - padding;

                                const int input_idx = (ich * in_height + ih) * in_width + iw;
                                const int weight_idx = ((och * icpg + ic) * kernel_size + kh) * kernel_size + kw;

                                sum += (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) ? 0.f :
                                        input[input_idx] * weight[weight_idx];
                            }
                        }
                    }

                    const int output_idx = (och * out_height + oh) * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
    delete[] weight;
    delete[] bias;
}
