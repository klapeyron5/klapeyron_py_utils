from numbers import Integral


def is_any_int(x):
    """
    Checks if x is any integer type
    """
    return isinstance(x, Integral)


def is_iterable(x):
    """
    Checks if x is iterable
    """
    try:
        iter(x)
        return True
    except TypeError:
        return False
    except Exception:
        raise Exception
