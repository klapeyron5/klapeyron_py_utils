import numpy as np
from klapeyron_py_utils.data_visualization.utils import plot_1D
import scipy.signal
import math


def generate_signal(freqs, ampls, fs=20, T=3):
    """
    :param freqs:
    :param ampls:
    :param fs: sample frequency
    :param T: whole time of signal
    :return:
    """
    assert len(freqs) == len(ampls)
    t = np.arange(0, T, 1/fs)
    assert t.shape[0] == fs * T
    s = 0
    for f, a in zip(freqs, ampls):
        s += a*np.sin(2*np.pi*f*t)
    return s, t, fs


def process_signal(s, t, fs=20, title='', fr_cut=None):
    print('--'+title)
    plot_1D(x=t, y=s, xlabel='time (s)', title=title,)

    sp = np.abs(np.fft.fft(s))
    sp = sp[:len(sp)//2]
    sp_fr = fs/2*np.linspace(0, 1, len(sp))
    max_fr = sp_fr[np.argmax(sp)]
    print('predicted from full spectrum: '+str(max_fr))
    plot_1D(x=sp_fr, y=sp, marker='o', markersize=3, color='b', linestyle='None', title=title,
            xlabel='Hz')
    if fr_cut is not None:
        assert len(fr_cut) == 2
        l = np.where(sp_fr >= fr_cut[0])[0][0]
        r = np.where(sp_fr <= fr_cut[1])[0][-1]

        sp_fr_cut = sp_fr[l:r]
        sp_cut = sp[l:r]
        max_fr = sp_fr_cut[np.argmax(sp_cut)]
        print('predicted from cut spectrum: '+str(max_fr))
    pass


lowest_bpm = 40
highest_bpm = 240
lowest_bps = lowest_bpm / 60
highest_bps = highest_bpm / 60

freqs = np.random.uniform(0, 10, 100)
freqs = np.append(freqs, [1])
ampls = np.random.uniform(0, 1, 100)
ampls = np.append(ampls, [3.0])
print('Answer: '+str(freqs[np.argmax(ampls)]))
# s, t, fs = generate_signal(freqs=[1,3,4], ampls=[2,1,3])
s, t, fs = generate_signal(freqs=freqs, ampls=ampls)
process_signal(s, t[:len(s)], fs, title='source', fr_cut=[lowest_bps, highest_bps])

print('filter: ['+str(lowest_bps)+', '+str(highest_bps)+'] Hz')
n_f = 7
f = scipy.signal.firwin(n_f, cutoff=[lowest_bps, highest_bps], fs=20)

s_f = np.convolve(s, f, mode='valid')
process_signal(s_f, t[:len(s_f)], fs, title='filtered valid', fr_cut=[lowest_bps, highest_bps])

s_f = np.convolve(s, f, mode='same')
process_signal(s_f, t[:len(s_f)], fs, title='filtered same', fr_cut=[lowest_bps, highest_bps])