import numpy
import torch
from path import Path

import numpy as np
import os
import struct

def convert():
    fusionnet_test_weights = "/home/nhsmt1123/master-thesis/deep-video-mvs/dvmvs/fusionnet/weights"
    checkpoints = sorted(Path(fusionnet_test_weights).files())

    base_dir = os.path.dirname(os.path.abspath(__file__)) / Path("../params")

    fp = open(base_dir / "params", "wb")
    fn = open(base_dir / "n_params", "wb")

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
            d = bytearray()
            for v in param:
                d += struct.pack('f', v)
            fp.write(d)
            fn.write(struct.pack('i', len(param)))
            cnt += len(param)
        print(cnt)
        print(len(params_out))

    fp.close()
    fn.close()


if __name__ == '__main__':
    convert()
