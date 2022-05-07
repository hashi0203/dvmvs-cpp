#pragma once
#include "config.h"
#include "torch.h"

void Conv2d(const float* input,
            float* output,
            const float *params,
            unordered_map<string, int> mp,
            const string param_path,
            const int in_channels, const int in_height, const int in_width,
            const int out_channels, const int out_height, const int out_width,
            const int kernel_size, const int stride, const int padding, const int groups, const bool apply_bias) {

    // https://ichi.pro/conv-2-d-saigo-ni-fuxowa-do-pasu-de-nani-ga-okoru-ka-o-rikaisuru-30488625459528
    if (mp.find(param_path + ".weight") == mp.end())
        cout << param_path + ".weight" << "\n";
    if (mp.find(param_path + ".bias") == mp.end() && apply_bias)
        cout << param_path + ".bias" << "\n";

    const float* weight = params + mp[param_path + ".weight"];
    const float* bias = params + mp[param_path + ".bias"]; // maybe invalid pointer

    const int ocpg = out_channels / groups;
    const int icpg = in_channels / groups;
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < ocpg; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = 0.f;
                    const int och = g * ocpg + oc;

                    for (int ic = 0; ic < icpg; ic++) {
                        const int ich = g * icpg + ic;

                        for (int kh = 0; kh <= 2*padding; kh++) {
                            for (int kw = 0; kw <= 2*padding; kw++) {
                                const int ih = oh * stride + kh - padding;
                                const int iw = ow * stride + kw - padding;

                                const int input_idx = (ich * in_height + ih) * in_width + iw;
                                const int weight_idx = ((och * icpg + ic) * kernel_size + kh) * kernel_size + kw;

                                sum += (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) ? 0 :
                                        input[input_idx] * weight[weight_idx];
                            }
                        }
                    }

                    sum += (apply_bias) ? bias[och] : 0.f;
                    const int output_idx = (och * out_height + oh) * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}


// template <int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int kernel_size, int stride, int padding, int groups, bool apply_bias=false>
// class Conv2d{
// public:
//     // Conv2d() {
//     //     kaiming_uniform_<out_channels, in_channels / groups, kernel_size>(sqrt(5), weight);
//     //     // if self.bias is not None:
//     //     //     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
//     //     //     bound = 1 / math.sqrt(fan_in)
//     //     //     init.uniform_(self.bias, -bound, bound)
//     // }

//     // float weight[out_channels][in_channels / groups][kernel_size][kernel_size];
//     float ****weight = new float***[out_channels];
//     float *bias = new float[out_channels];

//     Conv2d(const string param_path) : param_path(param_path) {
//         new_4d(weight, out_channels, in_channels / groups, kernel_size, kernel_size);
//         // load parameters
//         ifstream ifs;
//         ifs = open_file(param_path + ".weight");
//         for (int i = 0; i < out_channels; i++)  for (int j = 0; j < in_channels / groups; j++) for (int k = 0; k < kernel_size; k++)
//             ifs.read((char*) weight[i][j][k], sizeof(float) * kernel_size);

//         if (apply_bias) {
//             ifs = open_file(param_path + ".bias");
//             ifs.read((char*) bias, sizeof(float) * out_channels);
//         } else {
//             for (int i = 0; i < out_channels; i++) bias[i] = 0;
//         }
//     }

//     void forward(const float input[in_channels][in_height][in_width], float output[out_channels][out_height][out_width]) {
//         // https://ichi.pro/conv-2-d-saigo-ni-fuxowa-do-pasu-de-nani-ga-okoru-ka-o-rikaisuru-30488625459528
//         const int ocpg = out_channels / groups;
//         const int icpg = in_channels / groups;
//         for (int g = 0; g < groups; g++) {
//             for (int oc = 0; oc < ocpg; oc++) {
//                 for (int oh = 0; oh < out_height; oh++) {
//                     for (int ow = 0; ow < out_width; ow++) {
//                         float sum = 0.f;
//                         int och = g * ocpg + oc;

//                         for (int ic = 0; ic < icpg; ic++) {
//                             int ich = g * icpg + ic;

//                             for (int kh = 0; kh <= 2*padding; kh++) {
//                                 for (int kw = 0; kw <= 2*padding; kw++) {
//                                     int ih = oh * stride + kh - padding;
//                                     int iw = ow * stride + kw - padding;

//                                     sum += (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) ? 0 : weight[och][ic][kh][kw] * input[ich][ih][iw];
//                                 }
//                             }
//                         }

//                         sum += bias[och];
//                         output[och][oh][ow] = sum;
//                     }
//                 }
//             }
//         }
//         close();
//     }

//     void close() {
//         delete_4d(weight, out_channels, in_channels / groups, kernel_size, kernel_size);
//     }

//     // void forward_org(float input[in_channels][in_height][in_width], float output[out_channels][out_height][out_width]) {
//     //     float padded_input[in_channels][in_height+2*padding][in_width+2*padding];
//     //     pad_input<in_channels, in_height, in_width, padding>(input, padded_input);
//     //     for (int oc = 0; oc < out_channels; oc++) for (int oh = 0; oh < out_height; oh++) for (int ow = 0; ow < out_width; ow++)
//     //         output[oc][oh][ow] = 0;

//     //     for (int oc = 0; oc < out_channels; oc++) {
//     //         for (int ic = 0; ic < in_channels; ic++) {
//     //             for (int oh = 0; oh < out_height; oh++) {
//     //                 for (int ow = 0; ow < out_width; ow++) {
//     //                     for (int ph = 0; ph <= 2*padding; ph++) {
//     //                         for (int pw = 0; pw <= 2*padding; pw++) {
//     //                             output[oc][oh][ow] += weight[oc][ic][ph][pw] * padded_input[ic][oh*stride+ph][ow*stride+pw];
//     //                         }
//     //                     }
//     //                 }
//     //             }
//     //         }
//     //     }
//     // }

// private:
//     string param_path;
//     // const int dilation, groups, bias, padding_mode;
//     // self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
// };
