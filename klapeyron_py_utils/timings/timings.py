from time import time
from klapeyron_py_utils.types import common_types


def get_time(f, n_avg=100, warmup=True, **f_kwargs):
    """
    Measures working time of any function
    :param f: function to be measured
    :param n_avg: number of times to average single function execution
    :param warmup: blank launch of f befor time measuring
    :param f_kwargs: measured function args to pass
    :return: n_avg times averaged execution time of function f
    """
    assert hasattr(f, '__call__')
    common_types.is_any_int(n_avg)
    assert n_avg > 0
    if warmup: f(**f_kwargs)
    t = time()
    for _ in (f(**f_kwargs) for _ in range(n_avg)): pass
    wasted_time = (time() - t) / n_avg
    return wasted_time
