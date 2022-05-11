import torch
import numpy as np
import os
from path import Path
import struct

INTMAX = [None, 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767]

def quantize(param, bit):
    scale = float(INTMAX[bit] / max(-np.min(param), np.max(param)))
    shift = int(np.floor(np.log2(scale)))
    scaled_param = param * (2 ** shift)
    return scale, shift, np.round(scaled_param)

def main():
    print("%38s: (  scale, 2^shift, shift,   min,   max)" % "output")
    print("-" * (38 + 41))

    fusionnet_test_weights = "/home/nhsmt1123/master-thesis/deep-video-mvs/dvmvs/fusionnet/weights"
    checkpoints = sorted(Path(fusionnet_test_weights).files())

    base_dir = os.path.dirname(os.path.abspath(__file__)) / Path("../params")
    fp = open(base_dir / "params_quantized", "wb")
    fv = open(base_dir / "values_quantized", "wb")
    fs = open(base_dir / "shifts_quantized", "wb")

    cnts = []
    offsets = []
    for checkpoint in checkpoints:
        params = torch.load(checkpoint)

        with open(base_dir / "files" / checkpoint.name, 'r') as f:
            files = f.read().split()

        cnt = 0
        for i, key in enumerate(files):
            if ".running_mean" in key:
                param = params[key].to('cpu').detach().numpy().copy().reshape(-1)
                rv = 1.0 / np.sqrt(params[files[i+1]].to('cpu').detach().numpy().copy().reshape(-1) + 1e-5)
                param *= rv
                bit = 12
            elif ".running_var" in key:
                param = rv
                bit = 16
            else: # ".weight" or ".bias"
                param = params[key].to('cpu').detach().numpy().copy().reshape(-1)
                bit = 10

            scale, shift, scaled_param = quantize(param, bit)

            if ".running_var" in key:
                shift += 1
                if shift > 0:
                    scaled_param = np.round(param * (2 ** shift))
                    offset = ((np.max(scaled_param) + np.min(scaled_param) + 1) / 2).astype("int")
                    scaled_param -= offset
                else:
                    offset = np.ceil((np.max(scaled_param) + np.min(scaled_param)) / 2).astype("int")
                    scaled_param = np.round((param - offset) * (2 ** shift))
                offsets.append(offset)

            data = scaled_param.astype("int16")
            d = bytearray()
            for v in data:
                d += struct.pack('h', v)
            fp.write(d)
            fv.write(struct.pack('i', len(data)))
            fs.write(struct.pack('i', shift))
            cnt += len(data)
            if (len(key) > 38):
                print("%s:" % key)
                print("%38s  (%7d, %7d, %5d, %5d, %5d)" % (" ", scale, 2**shift, shift, np.min(scaled_param), np.max(scaled_param)))
            else:
                print("%38s: (%7d, %7d, %5d, %5d, %5d)" % (key, scale, 2**shift, shift, np.min(scaled_param), np.max(scaled_param)))
        cnts.append(cnt)

        # savedir = 'npz'
        # os.makedirs(savedir, exist_ok=True)
        # for key in params:
        #     params[key] = params[key].to('cpu').detach().numpy().copy()

        # params["shifts"] = np.array(shifts)

        # np.savez_compressed(os.path.join(savedir, "params_quantized"), **params)

    print(cnts)
    print(len(offsets))
    print(offsets)

    fp.close()
    fv.close()
    fs.close()

if __name__ == '__main__':
    main()
