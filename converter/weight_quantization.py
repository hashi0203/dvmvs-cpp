import torch
import numpy as np
import os
from path import Path
import struct

def quantize(param):
    scale = float(32767 / max(-np.min(param), np.max(param)))
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
    for checkpoint in checkpoints:
        params = torch.load(checkpoint)

        # shifts = {}
        # for key in params:
        #     if params[key].dtype == torch.float32:
        #         scale, shift, scaled_param = quantize(params[key])
        #         params[key] = scaled_param
        #         shifts[key] = shift
        #         print("%15s: (%7d, %7d, %5d, %5d, %5d)" % (key, scale, 2**shift, shift, min(scaled_param.reshape(-1)), max(scaled_param.reshape(-1))))

        # torch.save(params, 'model_quantized.pt')

        with open(base_dir / "files" / checkpoint.name, 'r') as f:
            files = f.read().split()

        cnt = 0
        for key in files:
            if "running_var" in key: print(np.min(params[key].to('cpu').detach().numpy().copy().reshape(-1)))
            scale, shift, scaled_param = quantize(params[key].to('cpu').detach().numpy().copy().reshape(-1))
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

    fp.close()
    fv.close()
    fs.close()

if __name__ == '__main__':
    main()
