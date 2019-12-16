from numbers import Integral


def is_any_int(x):
    """
    Asserts x is any integer type
    """
    return isinstance(x,Integral)
