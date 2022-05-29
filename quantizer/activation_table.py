import numpy as np

def celu(x):
    return np.where(x > 0, x, 0) + np.where(x > 0, 0, np.exp(x) - 1)


def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def quantize(x, shift):
    return np.round(x * (1 << shift)).astype('int')


if __name__ == '__main__':
    scale = 3
    xbit = 8 # 8 でもまあいい
    sigshift = 14
    celushift = 12

    x = np.arange(1 << xbit) / (1 << xbit) * (1 << scale)
    print("constexpr int tbbit = %d;" % xbit)
    print("constexpr int tbshift = %d;" % (xbit - scale))
    print("constexpr int celushift = %d;" % celushift)
    print("constexpr int sigshift = %d;" % sigshift)

    c = quantize(celu(-x), celushift)
    print("constexpr qaint celu_table[%d] = {%s};" % (1 << xbit, ", ".join(map(str, c))))

    s = quantize(Sigmoid(x), sigshift)
    print("constexpr qaint Sigmoid_table[%d] = {%s};" % (1 << xbit, ", ".join(map(str, s))))
