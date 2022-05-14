import numpy as np

def celu(x):
    return np.where(x > 0, x, 0) + np.where(x > 0, 0, np.exp(x) - 1)


def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def quantize(x, bit):
    return np.round(x * (1 << (bit-1))).astype('int')


if __name__ == '__main__':
    scale = 6.0
    xbit = 10 # 8 でもまあいい
    ybit = 16

    x = np.arange(1 << xbit) / (1 << xbit) * 6.0
    print(x)
    print("constexpr int tbbit = %d;" % xbit)
    print("constexpr int actbit = %d;" % ybit)

    c = quantize(celu(-x), ybit)
    print("constexpr qaint celu_table[%d] = {%s};" % (1 << xbit, ", ".join(map(str, c))))

    s = quantize(Sigmoid(x), ybit)
    print("constexpr qaint Sigmoid_table[%d] = {%s};" % (1 << xbit, ", ".join(map(str, s))))
