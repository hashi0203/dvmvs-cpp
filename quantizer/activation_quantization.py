import cv2
import numpy as np
import torch
from path import Path
from tqdm import tqdm
import os
import struct

from config import Config
from dataset_loader import PreprocessImage, load_image
from model import FeatureExtractor, FeatureShrinker, CostVolumeEncoder, LSTMFusion, CostVolumeDecoder
from keyframe_buffer2 import KeyframeBuffer
from utils import cost_volume_fusion, get_non_differentiable_rectangle_depth_estimation, get_warp_grid_for_cost_volume_calculation

INTMAX = [None, 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767]
INTMIN = [None, -1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048, -4096, -8192, -16384, -32768]

# def quantize(f, act, bit, alpha=0.99):
#     act = np.abs(act)
#     act = np.sort(act)
#     idx = int(round(len(act) * alpha))
#     scale = float(INTMAX[bit] / act[idx])
#     shift = int(np.floor(np.log2(scale)))
#     f.write(struct.pack('i', shift))
#     return act[idx], scale, shift

def quantize(act, params, param_idx, scales, bit, alpha=0.99):
    data = act[1]
    if act[0] == "add":
        assert len(data) == 2
        pass
    elif act[0] == "cat":
        assert len(data) == 2 or len(data) == 3
        pass
    elif act[0] == "conv":
        assert len(data) == 2
        assert param_idx < len(params[0])
        weight, bias = params[0][param_idx]
        kernel_size, stride, padding, groups = params[1][param_idx]
        param_idx += 1

        x, y = data
        print(x.shape, y.shape)
        conv = torch.nn.Conv2d(x.shape[2], y.shape[2], kernel_size, padding=padding, stride=stride, groups=groups)
        # if param_idx == 1: print(conv.weight[0])
        conv.weight = torch.nn.Parameter(torch.tensor(weight))
        conv.bias = torch.nn.Parameter(torch.tensor(bias))
        pass
    elif act[0] == "sigmoid":
        assert len(data) == 0
        pass
    else:
        print(act[0])

    return param_idx


def predict():
    base_dir = os.path.dirname(os.path.abspath(__file__)) / Path("..")

    print("reading activation file...")
    npz_acts = np.load(base_dir / "params" / "acts.npz", allow_pickle=True)
    acts = npz_acts["acts"]

    print("reading param file...")
    npz_params = np.load(base_dir / "params" / "params.npz", allow_pickle=True)
    params = npz_params["params"].reshape(-1, 2)
    scales = npz_params["scales"]

    print(base_dir / "params" / "conv_params.txt")
    with open(base_dir / "params" / "conv_params.txt", "r") as f:
        params = (params, np.array(list(map(int, f.read().split()))).reshape(-1, 4))
    bit = 20

    f = open(base_dir / "params" / "actshifts_quantized", "wb")
    param_idx = 0
    # for i in range(n_acts):
    for act in acts:
    # for act in [("conv", [np.zeros(30000).reshape(3, -1), np.zeros(3200).reshape(32, -1)])]:
        # act = npz_acts["acts"][i]
        # print(len(act[1]), act[1][0].shape)
        param_idx = quantize(act, params, param_idx, scales, bit)
        # break
        # scale, quantize(act)
        # print(act[0])
        # print(quantize(f, act, 16))
    f.close()
    print(param_idx)
    # print(len(acts))


if __name__ == '__main__':
    predict()
