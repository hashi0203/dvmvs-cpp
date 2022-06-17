import numpy as np
from path import Path
import os
import struct

def compute_errors(gt, pred):
    # MSE, RMSE
    valid1 = gt >= 0.5
    valid2 = gt <= 20.0
    valid = valid1 & valid2
    gt = gt[valid]
    pred = pred[valid]

    if len(gt) == 0:
        return np.nan, np.nan

    differences = gt - pred
    squared_differences = np.square(differences)
    mse = np.mean(squared_differences)
    rmse = np.sqrt(mse)

    return mse, rmse


if __name__ == '__main__':
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    test_dataset_names = ["chess-seq-01", "chess-seq-02", "fire-seq-01", "fire-seq-02", "heads-seq-02", "office-seq-01", "office-seq-03", "redkitchen-seq-01", "redkitchen-seq-07"]
    for name in ["noqt", "qt"]:
        mses_npz, rmses_npz = {}, {}
        for test_dataset_name in test_dataset_names:
            files = sorted((base_dir / name / 'results' / test_dataset_name).files("*.bin"))
            predictions = []
            for file in files:
                data = []
                with open(file, 'rb') as f:
                    while True:
                        d = f.read(4)
                        if len(d) != 4:
                            break
                        data.append(struct.unpack('f', d))
                predictions.append(np.array(data).reshape(64, 96))

            gts = np.load(base_dir / "python/gts.npz")[test_dataset_name]
            print(name + "/" + test_dataset_name)
            assert len(predictions) == len(gts)

            mses, rmses = [], []
            for i, prediction in enumerate(predictions):
                mse, rmse = compute_errors(gts[i], prediction)
                print('%s: %.3f, %.3f' % (files[i].split('/')[-1][:-4], mse, rmse))

                mses.append(mse)
                rmses.append(rmse)

            print(test_dataset_name, "MSE", np.mean(mses))
            print(test_dataset_name, "RMSE", np.mean(rmses))

            mses_npz[test_dataset_name] = mses
            rmses_npz[test_dataset_name] = rmses

        np.savez_compressed(base_dir / (name + '-mses'), **mses_npz)
        np.savez_compressed(base_dir / (name + '-rmses'), **rmses_npz)

    mses_npz, rmses_npz = {}, {}
    for test_dataset_name in test_dataset_names:
        files = base_dir / 'depths' / test_dataset_name + ".npz"
        predictions = np.load(files)['depths'][:,0,0,:,:]

        gts = np.load(base_dir / "python/gts.npz")[test_dataset_name]
        assert len(predictions) == len(gts)

        mses, rmses = [], []
        for i, prediction in enumerate(predictions):
            mse, rmse = compute_errors(gts[i], prediction)
            print('%3d: %.3f, %.3f' % (i, mse, rmse))

            mses.append(mse)
            rmses.append(rmse)

        print(test_dataset_name, "MSE", np.mean(mses))
        print(test_dataset_name, "RMSE", np.mean(rmses))

        mses_npz[test_dataset_name] = mses
        rmses_npz[test_dataset_name] = rmses

    np.savez_compressed(base_dir / ('pynq-mses'), **mses_npz)
    np.savez_compressed(base_dir / ('pynq-rmses'), **rmses_npz)
