#pragma once
#include "settings.h"
#include "torch.h"

template <int in_channels, int in_height, int in_width, int out_channels, int out_height, int out_width, int kernel_size, int stride, int padding, int groups>
class Conv2d{
public:
    // Conv2d() {
    //     kaiming_uniform_<out_channels, in_channels / groups, kernel_size>(sqrt(5), weight);
    //     // if self.bias is not None:
    //     //     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    //     //     bound = 1 / math.sqrt(fan_in)
    //     //     init.uniform_(self.bias, -bound, bound)
    // }

    float weight[out_channels][in_channels / groups][kernel_size][kernel_size];

    Conv2d(const string param_path) : param_path(param_path) {
        // load parameters
        ifstream ifs = open_file(param_path + ".weight");
        ifs.read((char*) weight, sizeof(float) * out_channels * (in_channels / groups) * kernel_size * kernel_size);
    }

    void forward(float input[in_channels][in_height][in_width], float output[out_channels][out_height][out_width]) {
        // if self.padding_mode != 'zeros':
        //     return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
        //                     weight, self.bias, self.stride,
        //                     _pair(0), self.dilation, self.groups)

        float padded_input[in_channels][in_height+2*padding][in_width+2*padding];
        pad_input<in_channels, in_height, in_width, padding>(input, padded_input);
        for (int oc = 0; oc < out_channels; oc++) for (int oh = 0; oh < out_height; oh++) for (int ow = 0; ow < out_width; ow++)
            output[oc][oh][ow] = 0;

        // https://ichi.pro/conv-2-d-saigo-ni-fuxowa-do-pasu-de-nani-ga-okoru-ka-o-rikaisuru-30488625459528
        const int ocpg = out_channels / groups;
        const int icpg = in_channels / groups;
        for (int g = 0; g < groups; g++) {
            for (int oc = 0; oc < ocpg; oc++) {
                for (int ic = 0; ic < icpg; ic++) {
                    for (int oh = 0; oh < out_height; oh++) {
                        for (int ow = 0; ow < out_width; ow++) {
                            for (int ph = 0; ph <= 2*padding; ph++) {
                                for (int pw = 0; pw <= 2*padding; pw++) {
                                    output[ocpg * g + oc][oh][ow] += weight[ocpg * g + oc][ic][ph][pw] * padded_input[icpg * g + ic][oh*stride+ph][ow*stride+pw];
                                }
                            }
                        }
                    }
                }
            }
        }

    }

    // void forward_org(float input[in_channels][in_height][in_width], float output[out_channels][out_height][out_width]) {
    //     float padded_input[in_channels][in_height+2*padding][in_width+2*padding];
    //     pad_input<in_channels, in_height, in_width, padding>(input, padded_input);
    //     for (int oc = 0; oc < out_channels; oc++) for (int oh = 0; oh < out_height; oh++) for (int ow = 0; ow < out_width; ow++)
    //         output[oc][oh][ow] = 0;

    //     for (int oc = 0; oc < out_channels; oc++) {
    //         for (int ic = 0; ic < in_channels; ic++) {
    //             for (int oh = 0; oh < out_height; oh++) {
    //                 for (int ow = 0; ow < out_width; ow++) {
    //                     for (int ph = 0; ph <= 2*padding; ph++) {
    //                         for (int pw = 0; pw <= 2*padding; pw++) {
    //                             output[oc][oh][ow] += weight[oc][ic][ph][pw] * padded_input[ic][oh*stride+ph][ow*stride+pw];
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

private:
    string param_path;
    // const int in_channels, out_channels, kernel_size, stride, padding;
    // const int dilation, groups, bias, padding_mode;
    // const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    // const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    // float weight[out_channels, in_channels / groups, kernel_size];
    // self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
};
