import numpy as np
from path import Path
import os
import struct

INTMAX = [None, 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143, 524287, 1048575, 2097151, 4194303, 8388607, 16777215, 33554431, 67108863, 134217727, 268435455, 536870911, 1073741823, 2147483647]
INTMIN = [None, -1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048, -4096, -8192, -16384, -32768, -65536, -131072, -262144, -524288, -1048576, -2097152, -4194304, -8388608, -16777216, -33554432, -67108864, -134217728, -268435456, -536870912, -1073741824, -2147483648]

def quantize(act, bit, alpha=0.95):
    param = act[1].reshape(-1)
    param = np.abs(param)
    if act[0] == "add" or act[0] == "conv":
        param = np.sort(param)
        idx = int(round(len(param) * alpha))
        scale = float(INTMAX[bit-1] / param[idx])
        shift = int(np.floor(np.log2(scale)))
        return shift
    elif act[0] == "sigmoid" or act[0] == "celu":
        print("%7s: %.5f" % (act[0], np.max(param)))
        return None
    else:
        print(act[0])
        return None


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__)) / Path("..")

    print("reading activation file...")
    npz_acts = np.load(base_dir / "params" / "acts.npz", allow_pickle=True)
    acts = npz_acts["acts"]
    bit = 20

    print("quantizing...")
    f = open(base_dir / "params" / "act_shifts", "wb")
    cnt = 0
    for act in acts:
        shift = quantize(act, bit)
        if shift is not None:
            f.write(struct.pack('i', shift))
            cnt += 1

    f.close()
    print(cnt)
