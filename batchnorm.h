#pragma once

void BatchNorm2d(float* x, const string param_path,
                 const int channels, const int height, const int width) {

    // if (mp.find(param_path + ".running_mean") == mp.end())
    //     cout << param_path + ".running_mean" << "\n";
    // if (mp.find(param_path + ".running_var") == mp.end())
    //     cout << param_path + ".running_var" << "\n";
    // if (mp.find(param_path + ".weight") == mp.end())
    //     cout << param_path + ".weight" << "\n";
    // if (mp.find(param_path + ".bias") == mp.end())
    //     cout << param_path + ".bias" << "\n";

    // const float* running_mean = params + mp[param_path + ".running_mean"];
    // const float* running_var = params + mp[param_path + ".running_var"];
    // const float* weight = params + mp[param_path + ".weight"];
    // const float* bias = params + mp[param_path + ".bias"];

    const float* running_mean = params + start_idx[param_cnt++];
    const float* running_var = params + start_idx[param_cnt++];
    const float* weight = params + start_idx[param_cnt++];
    const float* bias = params + start_idx[param_cnt++];
    // print1(param_path + ".running_mean");
    // print1(param_path + ".running_var");
    // print1(param_path + ".weight");
    // print1(param_path + ".bias");

    // https://www.anarchive-beta.com/entry/2020/08/16/180000
    for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
        const int idx = (i * height + j) * width + k;
        const float xc = x[idx] - running_mean[i];
        const float xn = xc / sqrt(running_var[i] + 1e-5);
        x[idx] = weight[i] * xn + bias[i];
    }
}
