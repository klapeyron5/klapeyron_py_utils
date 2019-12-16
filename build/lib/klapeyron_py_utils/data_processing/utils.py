import numpy as np


def mov_avg(x, n=3, padding='valid'):
    """
    The logic is totally copied from
    https://stackoverflow.com/a/14314054/5030761
    :param x: signal to process
    :param n: moving window
    :return: trimmed x, processed by mov avg (trimmed for averaging window to be filled with existing values)
    """
    assert isinstance(x, np.ndarray)
    assert padding == 'valid'
    ret = np.cumsum(x, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n

    l = n//2
    r = -n+n//2+1
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
    :return: trimmed x, processed by mov avg (trimmed for averaging window to be filled with existing values)
    """
    assert isinstance(x, np.ndarray)
    assert padding == 'valid'
    ma, fitted_x = mov_avg(x, n)
    ret = []
    for e, m in zip(fitted_x, ma):
        ret.append(e/m)
    return ret
