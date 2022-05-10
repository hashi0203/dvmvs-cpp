import torch
from path import Path

import os
import struct

def convert():
    fusionnet_test_weights = "/home/nhsmt1123/master-thesis/deep-video-mvs/dvmvs/fusionnet/weights"
    checkpoints = sorted(Path(fusionnet_test_weights).files())

    for checkpoint in checkpoints:
        save_dir = Path("../params") / checkpoint.split("/")[-1]
        os.makedirs(save_dir, exist_ok=True)

        weights = torch.load(checkpoint)
        # f = open(save_dir + "_files", 'w')
        for key in weights:
            val = weights[key].to('cpu').detach().numpy().copy().reshape(-1)
            if val.dtype == 'float32':
                # f.write(key + '\n')
                d = bytearray()
                for v in val:
                    d += struct.pack('f', v)
                # open(save_dir / Path(key), 'wb').write(d)
            else:
                print(key)
            if "layer1.1." in key:
                print(key)
                print(val)
            # if key == "layer1.0.weight":
            #     print(val[:10])
            #     print(format(d[0], 'b'))
            #     print(d[:20])
            # else:
            #     print(val.dtype, key, checkpoint.split("/")[-1])
        # f.close()

if __name__ == '__main__':
    convert()
