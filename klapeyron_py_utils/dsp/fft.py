import numpy as np


def fft_np(s):
    assert len(s) % 2 == 0  # TODO
    sp = np.abs(np.fft.fft(s))
    sp = sp[:len(sp)//2]
    return sp
