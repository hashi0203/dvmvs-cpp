import numpy as np
from path import Path
import os
import struct
import matplotlib.pyplot as plt

INTMAX = [None, 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143, 524287, 1048575, 2097151, 4194303, 8388607, 16777215, 33554431, 67108863, 134217727, 268435455, 536870911, 1073741823, 2147483647]
INTMIN = [None, -1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048, -4096, -8192, -16384, -32768, -65536, -131072, -262144, -524288, -1048576, -2097152, -4194304, -8388608, -16777216, -33554432, -67108864, -134217728, -268435456, -536870912, -1073741824, -2147483648]

act_cnt = 0
base_dir = os.path.dirname(os.path.abspath(__file__)) / Path("..")
ln_aves = []
ln_inv_stds = []

def quantize(act, bit, alpha=0.95):
    global act_cnt, base_dir

    param = act[1].copy()
    if act[0] == "add":
        param.append(param[0] + param[1])

    param = [np.abs(p.reshape(-1)) for p in param]

    if act[0] in ["add", "conv", "interpolate", "relu"]:
        param = [np.sort(p) for p in param]
        idx = [int(round(len(p) * alpha)) for p in param]
        scale = [float(INTMAX[bit-1] / p[i]) for p, i in zip(param, idx)]
        shift = [int(np.floor(np.log2(s))) for s in scale]
        print(act[0], [p[i] for p, i in zip(param, idx)], shift)
        return shift
    elif act[0] in ["cost_volume", "cat", "layer_norm", "cell_hidden"]:
        param = [np.sort(p) for p in param]
        idx = [int(round(len(p) * alpha)) for p in param]
        scale = [float(INTMAX[bit-1] / p[i]) for p, i in zip(param, idx)]
        shift = [int(np.floor(np.log2(s))) for s in scale]
        print(act[0], [p[i] for p, i in zip(param, idx)], shift)
        if act[0] == "layer_norm":
            ln = act[1][-2].copy().transpose(2, 0, 1, 3, 4)
            ln = ln.reshape(ln.shape[0], ln.shape[1], -1)
            ln_ave = np.mean(ln, axis=2)
            ln_var = np.var(ln, axis=2)
            ln_aves.append(np.round(np.mean(ln_ave, axis=1) * (1 << shift[0])).astype('int32'))
            ln_inv_stds.append(np.round(1 / np.mean(np.sqrt(ln_var), axis=1) * (1 << shift[-1])).astype('int32'))
            # print(np.std(ln_ave[:10], axis=1))
            # print(np.std(ln_var[:10], axis=1))
        return shift
    elif act[0] in ["sigmoid", "celu"]:
        plt.figure()
        plt.hist(act[1][0].reshape(-1), bins=np.arange(-32, 35, 4))
        plt.savefig("%s/quantizer/%s_%d.png" % (base_dir, act[0], act_cnt))
        act_cnt += 1
        print("%7s: %.5f, %.5f" % (act[0], np.max(param[0]), np.max(param[1])))
        return None
    else:
        print(act[0])
        return None


if __name__ == '__main__':
    print("reading activation file...")
    npz_acts = np.load(base_dir / "params" / "acts.npz", allow_pickle=True)
    acts = npz_acts["acts"]
    bit = 16

    print("quantizing...")
    fs = [[open(base_dir / "params" / "cin_shifts", "wb"), open(base_dir / "params" / "cout_shifts", "wb")],
          [open(base_dir / "params" / "ain1_shifts", "wb"), open(base_dir / "params" / "ain2_shifts", "wb"), open(base_dir / "params" / "aout_shifts", "wb")],
          [open(base_dir / "params" / "oin_shifts", "wb"), open(base_dir / "params" / "oout_shifts", "wb")]]
    cnt = [0, 0, 0]
    shifts = []
    for act in acts:
        shift = quantize(act, bit)
        if act[0] in ["cost_volume", "cat", "layer_norm", "cell_hidden"]:
            shifts.append((act[0], shift))
        elif shift is not None:
            assert len(shift) == 2 or len(shift) == 3
            idx = 0 if act[0] == "conv" else 1 if act[0] == "add" else 2
            for s, f in zip(shift, fs[idx]):
                f.write(struct.pack('i', s))
            cnt[idx] += 1

    for i in range(len(fs)):
        for j in range(len(fs[i])):
            fs[i][j].close()
    print(cnt)
    print(shifts)

    print("constexpr int ln_idx[2] = {0, %d};" % (len(ln_aves[1])))
    print("constexpr qaint ln_aves[%d + %d] = {%s, %s};" % (len(ln_aves[0]), len(ln_aves[1]), ", ".join(map(str, ln_aves[0])), ", ".join(map(str, ln_aves[1]))))
    print("constexpr qaint ln_inv_stds[%d + %d] = {%s, %s};" % (len(ln_inv_stds[0]), len(ln_inv_stds[1]), ", ".join(map(str, ln_inv_stds[0])), ", ".join(map(str, ln_inv_stds[1]))))
