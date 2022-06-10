#pragma once

void Conv2d(const qaint* input,
            qaint* output,
            const string param_path,
            const int in_channels, const int in_height, const int in_width,
            const int out_channels, const int out_height, const int out_width,
            const int kernel_size, const int stride, const int padding, const int groups,
            const bool apply_scale, const string activation,
            const int act_in, int& act_out) {

    // https://ichi.pro/conv-2-d-saigo-ni-fuxowa-do-pasu-de-nani-ga-okoru-ka-o-rikaisuru-30488625459528

    // to check order of layers
    // print1(param_path + ".weight");
    // if (apply_bias) print1(param_path + ".bias");

    const int wshift = w_shifts[conv_cnt];
    const qwint* weight = weights + w_idx[conv_cnt];
    const int bshift = b_shifts[conv_cnt];
    const qbint* bias = biases + b_idx[conv_cnt];

    const int xshift = cin_shifts[conv_cnt];
    const int yshift = cout_shifts[conv_cnt];
    conv_cnt++;

    const int sshift = apply_scale ? s_shifts[bn_cnt] : 0;
    const qsint* scale = apply_scale ? scales + s_idx[bn_cnt++] : nullptr;

    print_neg_shift(param_path, "wshift", wshift);
    print_neg_shift(param_path, "bshift", bshift);
    print_neg_shift(param_path, "bshift > 32", 32 - bshift);
    print_neg_shift(param_path, "xshift", xshift);
    print_neg_shift(param_path, "yshift", yshift);
    if (apply_scale) print_neg_shift(param_path, "sshift", sshift);
    print_neg_shift(param_path, "bshift - (xshift + wshift)", bshift - (xshift + wshift));
    print_neg_shift(param_path, "xshift + wshift + sshift - yshift", xshift + wshift + sshift - yshift);

    if (nngen_code) {
        /*
        weight{act_cnt} = ng.variable(dtype=weight_dtype,
                                      shape=({out_channels}, {kernel_size}, {kernel_size}, {in_channels}),
                                      name={param_name} + ".weight")
        if (groups == 1) {
            weight{act_cnt}.set_value(params[{param_name} + ".weight"])
        } else {
            weight{act_cnt}_value = params[{param_name} + ".weight"]
            weigh{act_cnt}_value = np.zeros(({out_channels}, {kernel_size}, {kernel_size}, {in_channels}), dtype=np.int8)
            for i, j, k in np.ndindex(({out_channels}, {kernel_size}, {kernel_size})):
                weight{act_cnt}_value[i][j][k][i] = weight{act_cnt}_value_org[i][j][k][0]
            weight{act_cnt}.set_value(weight{act_cnt}_value)
        }

        if (bshift == 32) {
            if (apply_scale || activation != "none" || kernel_size >= 5) {
                printf("error: unexpected (apply_scale, activation, kernel_size) in conv without bias: (%s, %s, %d)).\n",
                       apply_scale ? "true" : "false", activation.c_str(), kernel_size);
            }

            rshift{act_cnt} = ng.constant([{xshift + wshift + sshift - yshift}], dtype=ng.int8)
            act{act_cnt} = ng.conv2d(act{act_in}, weight{act_cnt}, strides=(1, {stride}, {stride}, 1),
                                     rshift_out=rshift{act_cnt}, asymmetric_clip=True,
                                     par_ich=par_ich, par_och=par_ochs[({kernel_size}, {stride})],
                                     dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)
        } else {
            bias{act_cnt} = ng.variable(dtype=bias_dtype, shape=({out_channels},), name={param_name} + ".bias")
            if (bshift == (xshift + wshift)) {
                bias{act_cnt}.set_value(params[{param_name} + ".bias"])
            } else {
                bias{act_cnt}.set_value(np.round(params[{param_name} + ".bias"] / (float) (1 << {(bshift - (xshift + wshift))})).astype(params[{param_name} + ".bias"].dtype)
            }

            if (apply_scale) {
                scale{act_cnt} = ng.variable(dtype=scale_dtype, shape=({out_channels},), name={param_name} + ".scale")
                scale{act_cnt}.set_value(params[{param_name} + ".scale"])

                if (activation == "relu" && kernel_size <= 5) {
                    rshift{act_cnt} = ng.constant([{xshift + wshift + sshift - oout_shifts[other_cnt]}], dtype=ng.int8)
                    act{act_cnt} = ng.conv2d(act{act_in}, weight{act_cnt}, strides=(1, {stride}, {stride}, 1),
                                             bias=bias{act_cnt}, scale=scale{act_cnt}, rshift_out=rshift{act_cnt},
                                             act_func=ng.relu, asymmetric_clip=True,
                                             par_ich=par_ich, par_och=par_ochs[({kernel_size}, {stride})],
                                             dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)
                } else if (activation == "none" && kernel_size <= 5) {
                    rshift{act_cnt} = ng.constant([{xshift + wshift + sshift - yshift}], dtype=ng.int8)
                    act{act_cnt} = ng.conv2d(act{act_in}, weight{act_cnt}, strides=(1, {stride}, {stride}, 1),
                                             bias=bias{act_cnt}, scale=scale{act_cnt}, rshift_out=rshift{act_cnt},
                                             asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[({kernel_size}, {stride})],
                                             dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)
                } else {
                    printf("error: unexpected (activation, kernel_size) in conv with apply_scale: (%s, %d)\n", activation.c_str(), kernel_size);
                }
            } else {
                if (activation == "sigmoid" && kernel_size <= 5) {
                    rshift{act_cnt} = ng.constant([{xshift + wshift + sshift - tbshift}], dtype=ng.int8)
                    act{act_cnt} = ng.conv2d(act{act_in}, weight{act_cnt}, strides=(1, {stride}, {stride}, 1),
                                             bias=bias{act_cnt}, rshift_out=rshift{act_cnt},
                                             act_func=sigmoid, asymmetric_clip=True,
                                             par_ich=par_ich, par_och=par_ochs[({kernel_size}, {stride})],
                                             dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)
                } else if (activation == "none" && kernel_size <= 5) {
                    rshift{act_cnt} = ng.constant([{xshift + wshift + sshift - yshift}], dtype=ng.int8)
                    act{act_cnt} = ng.conv2d(act{act_in}, weight{act_cnt}, strides=(1, {stride}, {stride}, 1),
                                             bias=bias{act_cnt}, rshift_out=rshift{act_cnt},
                                             asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[({kernel_size}, {stride})],
                                             dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)
                } else {
                    printf("error: unexpected (activation, kernel_size) in conv without apply_scale: (%s, %d)\n", activation.c_str(), kernel_size);
                }
            }
        }
        */

        const char* param_name = param_path.c_str();
        printf("# [%d] conv\n", act_cnt);
        printf("weight%d = ng.variable(dtype=weight_dtype, shape=(%d, %d, %d, %d), name=\"%s.weight\")\n",
               act_cnt, out_channels, kernel_size, kernel_size, in_channels, param_name);
        if (groups == 1) {
            printf("weight%d.set_value(params[\"%s.weight\"])\n", act_cnt, param_name);
        } else {
            printf("weight%d_value_org = params[\"%s.weight\"]\n", act_cnt, param_name);
            printf("weight%d_value = np.zeros((%d, %d, %d, %d), dtype=np.int8)\n", act_cnt, out_channels, kernel_size, kernel_size, in_channels);
            printf("for i, j, k in np.ndindex((%d, %d, %d)):\n", out_channels, kernel_size, kernel_size);
            printf("\tweight%d_value[i][j][k][i] = weight%d_value_org[i][j][k][0]\n", act_cnt, act_cnt);
            printf("weight%d.set_value(weight%d_value)\n", act_cnt, act_cnt);
        }
        printf("\n");

        if (bshift == 32) {
            if (apply_scale || activation != "none" || kernel_size >= 5) {
                printf("error: unexpected (apply_scale, activation, kernel_size) in conv without bias: (%s, %s, %d)).\n",
                       apply_scale ? "true" : "false", activation.c_str(), kernel_size);
            }

            printf("rshift%d = ng.constant([%d], dtype=ng.int8)\n",
                   act_cnt, xshift + wshift + sshift - yshift);
            printf("act%d = ng.conv2d(act%d, weight%d, strides=(1, %d, %d, 1), rshift_out=rshift%d, "
                   "asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(%d, %d)], "
                   "dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)\n",
                   act_cnt, act_in, act_cnt, stride, stride, act_cnt, kernel_size, stride);
        } else {
            printf("bias%d = ng.variable(dtype=bias_dtype, shape=(%d,), name=\"%s.bias\")\n",
                   act_cnt, out_channels, param_name);
            if (bshift == (xshift + wshift)) {
                printf("bias%d.set_value(params[\"%s.bias\"])\n", act_cnt, param_name);
            } else {
                printf("bias%d.set_value(np.round(params[\"%s.bias\"] / (float) (1 << %d)).astype(params[\"%s.bias\"].dtype))\n",
                       act_cnt, param_name, bshift - (xshift + wshift), param_name);
            }
            printf("\n");

            if (apply_scale) {
                printf("scale%d = ng.variable(dtype=scale_dtype, shape=(%d,), name=\"%s.scale\")\n",
                    act_cnt, out_channels, param_name);
                printf("scale%d.set_value(params[\"%s.scale\"])\n", act_cnt, param_name);
                printf("\n");

                if (activation == "relu" && kernel_size <= 5) {
                    printf("rshift%d = ng.constant([%d], dtype=ng.int8)\n",
                           act_cnt, xshift + wshift + sshift - oout_shifts[other_cnt]);
                    printf("act%d = ng.conv2d(act%d, weight%d, strides=(1, %d, %d, 1), bias=bias%d, scale=scale%d, "
                           "rshift_out=rshift%d, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(%d, %d)], "
                           "dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)\n",
                           act_cnt, act_in, act_cnt, stride, stride, act_cnt, act_cnt, act_cnt, kernel_size, stride);
                } else if (activation == "none" && kernel_size <= 5) {
                    printf("rshift%d = ng.constant([%d], dtype=ng.int8)\n",
                           act_cnt, xshift + wshift + sshift - yshift);
                    printf("act%d = ng.conv2d(act%d, weight%d, strides=(1, %d, %d, 1), bias=bias%d, scale=scale%d, "
                           "rshift_out=rshift%d, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(%d, %d)], "
                           "dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)\n",
                           act_cnt, act_in, act_cnt, stride, stride, act_cnt, act_cnt, act_cnt, kernel_size, stride);
                } else {
                    printf("error: unexpected (activation, kernel_size) in conv with apply_scale: (%s, %d)\n", activation.c_str(), kernel_size);
                }
            } else {
                if (activation == "sigmoid" && kernel_size <= 5) {
                    printf("rshift%d = ng.constant([%d], dtype=ng.int8)\n",
                           act_cnt, xshift + wshift + sshift - tbshift);
                    printf("act%d = ng.conv2d(act%d, weight%d, strides=(1, %d, %d, 1), bias=bias%d, "
                           "rshift_out=rshift%d, act_func=sigmoid, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(%d, %d)], "
                           "dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)\n",
                           act_cnt, act_in, act_cnt, stride, stride, act_cnt, act_cnt, kernel_size, stride);
                } else if (activation == "none" && kernel_size <= 5) {
                    printf("rshift%d = ng.constant([%d], dtype=ng.int8)\n",
                           act_cnt, xshift + wshift + sshift - yshift);
                    printf("act%d = ng.conv2d(act%d, weight%d, strides=(1, %d, %d, 1), bias=bias%d, "
                           "rshift_out=rshift%d, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(%d, %d)], "
                           "dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)\n",
                           act_cnt, act_in, act_cnt, stride, stride, act_cnt, act_cnt, kernel_size, stride);
                } else {
                    printf("error: unexpected (activation, kernel_size) in conv without apply_scale: (%s, %d)\n", activation.c_str(), kernel_size);
                }
            }
        }
        printf("\n\n");

        act_out = act_cnt++;
    }

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

                    sum += bias[och] >> (bshift - (xshift + wshift));
                    sum = apply_scale ? sum * scale[och] : sum;
                    const int output_idx = (och * out_height + oh) * out_width + ow;
                    output[output_idx] = clip(sum >> (xshift + wshift + sshift - yshift));
                }
            }
        }
    }

    if (shift_ckeck) print1(yshift);

    if (activation == "relu") {
        ReLU(output, out_channels, out_height, out_width);
    } else if (activation == "sigmoid") {
        Sigmoid(output, out_channels, out_height, out_width);
    } else if (activation != "none") {
        print1(activation);
    }
}
