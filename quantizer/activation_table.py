import numpy as np

def celu(x):
    return np.where(x > 0, x, 0) + np.where(x > 0, 0, np.exp(x) - 1)


def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def quantize(x, shift):
    return np.round(x * (1 << shift)).astype('int')


if __name__ == '__main__':
    scale = 3
    xbit = 10 # 8 でもまあいい
    yshift = 20

    x = np.arange(1 << xbit) / (1 << xbit) * (1 << scale)
    print(x)
    print("constexpr int tbbit = %d;" % xbit)
    print("constexpr int tbshift = %d;" % (xbit - scale))
    print("constexpr int sigshift = %d;" % yshift)

    c = quantize(celu(-x), yshift)
    print("constexpr qaint celu_table[%d] = {%s};" % (1 << xbit, ", ".join(map(str, c))))

    s = quantize(Sigmoid(x), yshift)
    print("constexpr qaint Sigmoid_table[%d] = {%s};" % (1 << xbit, ", ".join(map(str, s))))
