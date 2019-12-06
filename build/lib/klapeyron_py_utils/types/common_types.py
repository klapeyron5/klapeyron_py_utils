from numbers import Integral


def assert_any_int(x):
    """
    Asserts x is any integer type
    """
    assert isinstance(x,Integral)
