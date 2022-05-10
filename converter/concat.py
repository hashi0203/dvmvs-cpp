from path import Path
import struct

def convert():
    checkpoints = sorted(Path("../params").dirs())

    fpp = open(Path("../params") / "params", "wb")
    fvv = open(Path("../params") / "values", "wb")

    for checkpoint in checkpoints:
        with open(Path("../params") / checkpoint.name + "_files", 'r') as f:
            params = f.read().split()
        # params = sorted(checkpoint.files())

        fp = open(Path("../params") / checkpoint.name + "_params", "wb")
        fk = open(Path("../params") / checkpoint.name + "_keys", "w")
        fv = open(Path("../params") / checkpoint.name + "_values", "wb")

        cnt = 0
        n_files = 0
        for param in params:
            with open(Path("../params") / checkpoint.name / param, "rb") as ft:
                data = ft.read()
            fp.write(data)
            fpp.write(data)
            fk.write(param + "\n")
            fv.write(struct.pack('i', len(data) // 4))
            fvv.write(struct.pack('i', len(data) // 4))
            cnt += len(data) // 4
            n_files += 1
        print("cnt: %d" % cnt)
        print("n_files: %d" % n_files)

        fp.close()
        fk.close()
        fv.close()

    fpp.close()
    fvv.close()

if __name__ == '__main__':
    convert()
