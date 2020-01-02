import numpy as np


def mov_avg(x, n=3, padding='valid'):
    """
    The logic of moving averaging realization is copied from https://stackoverflow.com/a/14314054/5030761
    :param x: 1D array to process
    :param n: moving window
    :param padding: guess x=[0,1,2,3,4,5], n=5
                    'valid': same x
                    'same_real': x=[3,4, 0,1,2,3,4,5, 1,2]
    :return: apply paddings to x, then mov_avg it with 'valid' paddings and return
    """
    x, fitted_x = pad(x, n, padding)

    ret = np.cumsum(x, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n

    assert len(fitted_x) == len(ret)
    return ret, fitted_x


def mov_norm(x, n=3, padding='valid'):
    """
    :param x: signal to process
    :param n: moving window
    :param padding: guess x=[0,1,2,3,4,5], n=5
                    'valid': same x
                    'same_real': x=[3,4, 0,1,2,3,4,5, 1,2]
    :return: trimmed x, processed by mov avg (trimmed for averaging window to be filled with existing values)
    """
    ma, fitted_x = mov_avg(x, n, padding)
    ret = []
    for e, m in zip(fitted_x, ma):
        ret.append(e/m)
    return ret


def pad(x, n=3, padding='valid'):
    """
    The logic is totally copied from
    https://stackoverflow.com/a/14314054/5030761
    :param x: signal to process
    :param n: moving window
    :param padding: 'valid', 'same_real'
    :return: padding=='valid':
                 same x
             padding=='same_real':
                 x=[3,4, 0,1,2,3,4,5, 1,2]
    """
    assert isinstance(x, np.ndarray)
    padding_valid = 'valid'
    padding_same_real = 'same_real'
    assert padding in {padding_valid, padding_same_real}

    l = n//2
    r = n-l-1

    if padding == padding_same_real:
        x = np.append(x[1+r:1+r+l], x, axis=0)
        x = np.append(x, x[-l-1-r:-l-1], axis=0)

    if r:
        fitted_x = x[l:-r]
    else:
        fitted_x = x[l:]
    return x, fitted_x


def __unittest_pad():
    x = np.array([0, 1, 2, 3, 4, 5])
    n=1
    xa = x.copy()
    y,_ = pad(x, n, padding='same_real')
    assert (xa==y).all()

    n=2
    xa = np.array([1]+list(x))
    y,_ = pad(x, n, padding='same_real')
    assert (xa==y).all()

    n=3
    xa = np.array([2]+list(x)+[3])
    y,_ = pad(x, n, padding='same_real')
    assert (xa==y).all()

    n=4
    xa = np.array([2,3]+list(x)+[2])
    y,_ = pad(x, n, padding='same_real')
    assert (xa==y).all()

    n=5
    xa = np.array([3,4]+list(x)+[1,2])
    y,_ = pad(x, n, padding='same_real')
    assert (xa==y).all()
