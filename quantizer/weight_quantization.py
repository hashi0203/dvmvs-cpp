import torch
import numpy as np
import os
from path import Path
import struct

INTMAX = [None, 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143, 524287, 1048575, 2097151, 4194303, 8388607, 16777215, 33554431, 67108863, 134217727, 268435455, 536870911, 1073741823, 2147483647]
INTMIN = [None, -1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048, -4096, -8192, -16384, -32768, -65536, -131072, -262144, -524288, -1048576, -2097152, -4194304, -8388608, -16777216, -33554432, -67108864, -134217728, -268435456, -536870912, -1073741824, -2147483648]

def quantize(param, bit):
    param = param.reshape(-1)
    m = max(-np.min(param), np.max(param))
    if m == 0:
        return np.inf, 0, param
    scale = float(INTMAX[bit] / m)
    shift = int(np.floor(np.log2(scale)))
    scaled_param = param * (2 ** shift)
    return scale, shift, np.round(scaled_param)

def quantize_save(params_out, bit, fps):
    cnt = 0
    for param in params_out:
        byte = (bit - 1) // 8 + 1
        byte = 4 if byte == 3 else byte
        _, shift, scaled_param = quantize(param, bit)
        scaled_param = scaled_param.astype('int%d' % (byte * 8))

        d = bytearray()
        fps[0].write(struct.pack('i', len(scaled_param)))
        fmt = [None, 'b', 'h', None, 'i']
        for p in scaled_param:
            d += struct.pack(fmt[byte], p)
        fps[1].write(d)
        fps[2].write(struct.pack('i', shift))
        cnt += len(scaled_param)
    return cnt


def main():
    fusionnet_test_weights = "/home/nhsmt1123/master-thesis/deep-video-mvs/dvmvs/fusionnet/weights"
    checkpoints = sorted(Path(fusionnet_test_weights).files())

    base_dir = os.path.dirname(os.path.abspath(__file__)) / Path("../params")
    fw = [open(base_dir / "n_weights", "wb"),
          open(base_dir / "weights_quantized", "wb"),
          open(base_dir / "weight_shifts", "wb")]
    wbit = 8

    fb = [open(base_dir / "n_biases", "wb"),
          open(base_dir / "biases_quantized", "wb"),
          open(base_dir / "bias_shifts", "wb")]
    bbit = 22

    fs = [open(base_dir / "n_scales", "wb"),
          open(base_dir / "scales_quantized", "wb"),
          open(base_dir / "scale_shifts", "wb")]
    sbit = 8

    weights_out = []
    biases_out = []
    scales_out = []
    for checkpoint in checkpoints:
        with open(base_dir / "files" / checkpoint.name, 'r') as f:
            files = f.read().split()

        weights = torch.load(checkpoint)
        params = [weights[key].cpu().detach().numpy().copy() for key in files]
        idx = 0
        while idx < len(files):
            if ".running_mean" in files[idx]:
                assert ".weight" in files[idx-1]

                running_mean = params[idx]
                running_var = params[idx+1]
                weight = params[idx+2]
                bias = params[idx+3]

                wrv = weight / np.sqrt(running_var + 1e-5)
                scales_out.append(wrv)
                biases_out.append(bias - running_mean * wrv)
                idx += 4
            elif ".weight" in files[idx]:
                weights_out.append(params[idx])
                idx += 1
            elif ".bias" in files[idx]:
                biases_out.append(params[idx])
                idx += 1

        if checkpoint.name == "3_lstm_fusion":
            biases_out.append(np.zeros(params[idx-1].shape[0]))

    wcnt = quantize_save(weights_out, wbit, fw)
    bcnt = quantize_save(biases_out, bbit, fb)
    scnt = quantize_save(scales_out, sbit, fs)

    print(len(weights_out), wcnt)
    print(len(biases_out), bcnt)
    print(len(scales_out), scnt)


if __name__ == '__main__':
    main()
