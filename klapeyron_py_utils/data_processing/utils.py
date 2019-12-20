import numpy as np


def mov_avg(x, n=3, padding='valid'):
    """
    The logic is totally copied from
    https://stackoverflow.com/a/14314054/5030761
    :param x: signal to process
    :param n: moving window
    :param padding: guess x=[0,1,2,3,4,5], n=5
                    'valid': same x
                    'same_real': x=[3,4, 0,1,2,3,4,5, 1,2]
    :return: trimmed x, processed by mov avg (trimmed for averaging window to be filled with existing values)
    """
    assert isinstance(x, np.ndarray)
    padding_valid = 'valid'
    padding_same_real = 'same_real'
    assert padding in {padding_valid, padding_same_real}

    l = n//2
    r = -n+n//2+1  # n-1-n//2

    if padding == padding_same_real:
        x = np.append(x[-r+1:n], x, axis=0)
        x = np.append(x, x[-n:r-1], axis=0)

    ret = np.cumsum(x, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n

    if r:
        fitted_x = x[l:r]
    else:
        fitted_x = x[l:]
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
