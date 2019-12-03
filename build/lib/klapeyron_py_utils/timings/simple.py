from time import process_time


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
    if warmup: f(**f_kwargs)
    t = process_time()
    for _ in (f(**f_kwargs) for _ in range(n_avg)): pass
    wasted_time = (process_time() - t) / n_avg
    return wasted_time
