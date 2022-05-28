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

    const int mshift = max(bshift, xshift + wshift + sshift);

    print_neg_shift(param_path, "wshift", wshift);
    print_neg_shift(param_path, "bshift", bshift);
    print_neg_shift(param_path, "xshift", xshift);
    print_neg_shift(param_path, "yshift", yshift);
    if (apply_scale) print_neg_shift(param_path, "sshift", sshift);
    print_neg_shift(param_path, "mshift - yshift", mshift - yshift);

    if (nngen_code) {
        /*
        weight{act_cnt} = ng.variable(dtype=weight_dtype,
                                    shape=({out_channels}, {kernel_size}, {kernel_size}, {in_channels}),
                                    name=param_name + ".weight")
        if groups == 1:
            weight{act_cnt}.set_value(params[{param_name} + ".weight"])
        else:
            weight{act_cnt}.set_value(np.array([[[[params[{param_name} + ".weight"][i][j][k][0] if i == l else 0 for l in range({in_channels})]
                                                for k in range({kernel_size})] for j in range({kernel_size})] for i in range({out_channels})]))

        bias{act_cnt} = ng.variable(dtype=bias_dtype, shape=({out_channels},), name=param_name + ".bias")
        bias{act_cnt}.set_value(params[param_name + ".bias"])

        if apply_scale:
            scale{act_cnt} = ng.variable(dtype=scale_dtype, shape=({out_channels},), name=param_name + ".scale")
            scale{act_cnt}.set_value(params[param_name + ".scale"])

            conv{act_cnt} = ng.multiply(ng.conv2d(act{act_in}, weight{act_cnt},
                                                strides=(1, {stride}, {stride}, 1),
                                                dtype=act_dtype,
                                                sum_dtype=ng.int32), scale{act_cnt})
        else:
            conv{act_cnt} = ng.conv2d(act{act_in}, weight{act_cnt},
                                    strides=(1, stride, stride, 1),
                                    dtype=act_dtype,
                                    sum_dtype=ng.int32)

        if bshift == xshift + wshift + sshift:
            sum{act_cnt} = ng.add(conv{act_cnt}, bias{act_cnt})
        elif bshift > xshift + wshift + sshift:
            lshift{act_cnt} = ng.constant([mshift - (xshift + wshift + sshift)], dtype=ng.int8)
            sum{act_cnt} = ng.add(ng.lshift(conv{act_cnt}, lshift{act_cnt}), bias{act_cnt})
        else:
            lshift{act_cnt} = ng.constant([mshift - bshift], dtype=ng.int8)
            sum{act_cnt} = ng.add(conv{act_cnt}, ng.lshift(bias{act_cnt}, lshift{act_cnt}))

        if activation == "relu":
            rshift{act_cnt} = ng.constant([mshift - oout_shifts[other_cnt]], dtype=ng.int8)
            act{act_cnt} = ng.relu(ng.rshift_round(sum{act_cnt}, rshift{act_cnt}))
        elif activation == "sigmoid":
            rshift{act_cnt} = ng.constant([mshift - tbshift], dtype=ng.int8)
            act{act_cnt} = ng.sigmoid(ng.rshift_round(sum{act_cnt}, rshift{act_cnt}), lut_addrwidth=9, lut_clip=8.0, range_rate=1.0)
        elif activation == "none":
            rshift{act_cnt} = ng.constant([mshift - yshift], dtype=ng.int8)
            act{act_cnt} = ng.rshift_round(sum{act_cnt}, rshift{act_cnt})
        */

        const char* param_name = param_path.c_str();
        printf("# [%d] conv\n", act_cnt);
        printf("weight%d = ng.variable(dtype=weight_dtype, shape=(%d, %d, %d, %d), name=\"%s.weight\")\n",
               act_cnt, out_channels, kernel_size, kernel_size, in_channels, param_name);
        if (groups == 1) {
            printf("weight%d.set_value(params[\"%s.weight\"])\n", act_cnt, param_name);
        } else {
            printf("weight%d.set_value(np.array([[[[params[\"%s.weight\"][i][j][k][0] if i == l else 0 for l in range(%d)] "
                                                "for k in range(%d)] for j in range(%d)] for i in range(%d)]))\n",
                   act_cnt, param_name, in_channels, kernel_size, kernel_size, out_channels);
        }
        printf("\n");

        printf("bias%d = ng.variable(dtype=bias_dtype, shape=(%d,), name=\"%s.bias\")\n",
               act_cnt, out_channels, param_name);
        printf("bias%d.set_value(params[\"%s.bias\"])\n", act_cnt, param_name);
        printf("\n");

        if (apply_scale) {
            printf("scale%d = ng.variable(dtype=scale_dtype, shape=(%d,), name=\"%s.scale\")\n",
                   act_cnt, out_channels, param_name);
            printf("scale%d.set_value(params[\"%s.scale\"])\n", act_cnt, param_name);
            printf("\n");

            printf("conv%d = ng.multiply(ng.conv2d(act%d, weight%d, strides=(1, %d, %d, 1), dtype=act_dtype, sum_dtype=ng.int32), scale%d)\n",
                   act_cnt, act_in, act_cnt, stride, stride, act_cnt);
        } else {
            printf("conv%d = ng.conv2d(act%d, weight%d, strides=(1, %d, %d, 1), dtype=act_dtype, sum_dtype=ng.int32)\n",
                   act_cnt, act_in, act_cnt, stride, stride);
        }
        printf("\n");

        if (bshift == xshift + wshift + sshift) {
            printf("sum%d = ng.add(conv%d, bias%d)\n", act_cnt, act_cnt, act_cnt);
        } else if (bshift > xshift + wshift + sshift) {
            printf("lshift%d = ng.constant([%d], dtype=ng.int8)\n",
                   act_cnt, mshift - (xshift + wshift + sshift));
            printf("sum%d = ng.add(ng.lshift(conv%d, lshift%d), bias%d)\n", act_cnt, act_cnt, act_cnt, act_cnt);
        } else {
            printf("lshift%d = ng.constant([%d], dtype=ng.int8)\n",
                   act_cnt, mshift - bshift);
            printf("sum%d = ng.add(conv%d, ng.lshift(bias%d, lshift%d))\n", act_cnt, act_cnt, act_cnt, act_cnt);
        }

        if (activation == "relu") {
            printf("rshift%d = ng.constant([%d], dtype=ng.int8)\n",
                   act_cnt, mshift - oout_shifts[other_cnt]);
            printf("act%d = ng.relu(ng.rshift_round(sum%d, rshift%d))\n",
                   act_cnt, act_cnt, act_cnt);
        } else if (activation == "sigmoid") {
            printf("rshift%d = ng.constant([%d], dtype=ng.int8)\n",
                   act_cnt, mshift - tbshift);
            printf("act%d = ng.sigmoid(ng.rshift_round(sum%d, rshift%d), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=ng.int16)\n",
                   act_cnt, act_cnt, act_cnt);
        } else if (activation == "none") {
            printf("rshift%d = ng.constant([%d], dtype=ng.int8)\n",
                   act_cnt, mshift - yshift);
            printf("act%d = ng.rshift_round(sum%d, rshift%d)\n",
                   act_cnt, act_cnt, act_cnt);
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

                    sum = apply_scale ? sum * scale[och] : sum;
                    sum <<= mshift - (xshift + wshift + sshift);
                    sum += bias[och] << (mshift - bshift);
                    const int output_idx = (och * out_height + oh) * out_width + ow;
                    output[output_idx] = sum >> (mshift - yshift);
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
