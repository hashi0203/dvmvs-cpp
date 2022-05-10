import os
from path import Path
import struct

def convert():
    base_dir = os.path.dirname(os.path.abspath(__file__)) / Path("../params")

    checkpoints = sorted(base_dir.dirs())
    fpp = open(base_dir / "params", "wb")
    fvv = open(base_dir / "values", "wb")

    for checkpoint in checkpoints:
        if checkpoint.name == "files":
            continue

        with open(base_dir / "files" / checkpoint.name, 'r') as f:
            params = f.read().split()

        cnt = 0
        n_files = 0
        for param in params:
            with open(base_dir / checkpoint.name / param, "rb") as ft:
                data = ft.read()
            fpp.write(data)
            fvv.write(struct.pack('i', len(data) // 4))
            cnt += len(data) // 4
            n_files += 1
        print("cnt: %d" % cnt)
        print("n_files: %d" % n_files)

    fpp.close()
    fvv.close()

if __name__ == '__main__':
    convert()
