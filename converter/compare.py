import numpy as np
from path import Path

from path import Path
import os
import sys

def fileopen(filepath):
    with open(filepath, 'r') as f:
        data = np.array(list(map(float, f.read().split())))
    return data

def compare_results(base_dir, i):
    filename = "%05d.txt" % i
    try:
        data1 = fileopen(base_dir / "results-qt" / filename)
        data2 = fileopen(base_dir / "../dvmvs-cpp2/results" / filename)
        print(filename + ":", np.corrcoef(data1, data2)[0, 1])
        print(np.mean(data1), np.std(data1))
        print(np.mean(data2), np.std(data2))
    except:
        pass

def compare_interm(base_dir, filename1, filename2):
    data1 = fileopen(base_dir / "results-qt" / filename1)
    data2 = fileopen(base_dir / "../dvmvs-cpp2/results" / filename2)
    print(filename1 + ", " + filename2 + ":", np.corrcoef(data1, data2)[0, 1])
    print(np.mean(data1), np.std(data1))
    print(np.mean(data2), np.std(data2))


def main(args):
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent

    assert len(args) <= 2

    if len(args) == 0:
        for i in range(9, 30):
            compare_results(base_dir, i)
    elif len(args) == 1:
        try:
            compare_results(base_dir, int(args[0]))
        except ValueError:
            compare_interm(base_dir, args[0], args[0])
    else:
        compare_interm(base_dir, args[0], args[1])


if __name__ == '__main__':
    main(sys.argv[1:])
