import torch
import numpy as np
import os
from path import Path
import struct

INTMAX = [None, 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767]
INTMIN = [None, -1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048, -4096, -8192, -16384, -32768]

def quantize(param, bit):
    scale = max(-np.min(param), np.max(param) * (-INTMIN[bit]) / INTMAX[bit])
    scaled_param = param if scale == 0 else param / scale * (-INTMIN[bit])
    return scale, np.round(scaled_param).astype("int16")

    # mm = max(-np.min(param), np.max(param))
    # scale = 1 if mm == 0 else float(INTMAX[bit] / mm)
    # shift = int(np.floor(np.log2(scale)))
    # scaled_param = param * (2 ** shift)
    # return scale, shift, np.round(scaled_param).astype("int16")

def main():
    print("(scale, min, max)")
    print("-----------------")

    fusionnet_test_weights = "/home/nhsmt1123/master-thesis/deep-video-mvs/dvmvs/fusionnet/weights"
    checkpoints = sorted(Path(fusionnet_test_weights).files())

    base_dir = os.path.dirname(os.path.abspath(__file__)) / Path("../params")
    fp = open(base_dir / "params_quantized", "wb")
    fn = open(base_dir / "n_params", "wb")
    fs = open(base_dir / "param_scales", "wb")
    # fs = open(base_dir / "param_shifts", "wb")

    cnts = []
    for checkpoint in checkpoints:
        with open(base_dir / "files" / checkpoint.name, 'r') as f:
            files = f.read().split()

        weights = torch.load(checkpoint)
        params = [weights[key].to('cpu').detach().numpy().copy() for key in files]
        params_out = []
        idx = 0
        while idx < len(files):
            if ".running_mean" in files[idx]:
                assert ".weight" in files[idx-1]

                running_mean = params[idx]
                running_var = params[idx+1]
                weight = params[idx+2]
                bias = params[idx+3]

                wrv = weight / np.sqrt(running_var + 1e-5)
                params_out[-1] *= wrv[:, None, None, None]
                params_out.append(bias - running_mean * wrv)
                idx += 4
            else:
                params_out.append(params[idx])
                idx += 1

        if checkpoint.name == "3_lstm_fusion":
            params_out.append(np.zeros(params_out[-1].shape[0]))

        cnt = 0
        for param in params_out:
            param = param.reshape(-1)
            bit = 10
            scale, scaled_param = quantize(param, bit)
            # scale, shift, scaled_param = quantize(param, bit)

            d = bytearray()
            for v in scaled_param:
                d += struct.pack('h', v)
            fp.write(d)
            fn.write(struct.pack('i', len(scaled_param)))
            fs.write(struct.pack('f', scale))
            # fs.write(struct.pack('i', shift))
            cnt += len(scaled_param)
            print("(%.3f, %5d, %5d)" % (scale, np.min(scaled_param), np.max(scaled_param)))
        cnts.append(cnt)

        # savedir = 'npz'
        # os.makedirs(savedir, exist_ok=True)
        # for key in params:
        #     params[key] = params[key].to('cpu').detach().numpy().copy()

        # params["shifts"] = np.array(shifts)

        # np.savez_compressed(os.path.join(savedir, "params_quantized"), **params)

    print(cnts)

    fp.close()
    fn.close()
    fs.close()

if __name__ == '__main__':
    main()
