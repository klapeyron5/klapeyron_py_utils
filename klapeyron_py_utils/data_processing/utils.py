import numpy as np


def mov_avg(x, n=3):
    """
    The function is totally copied from
    https://stackoverflow.com/a/14314054/5030761
    :param x: signal to process
    :param n: moving window
    :return: trimmed x, processed by mov avg (trimmed for averaging window to be filled with existing values)
    """
    assert isinstance(x, np.ndarray)
    ret = np.cumsum(x, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
