import matplotlib.pyplot as plt


def plot_1D(y, x=None, title=None, xlabel=None, ylabel=None, **kwargs):
    if x is None:
        plt.plot(y, **kwargs)
    else:
        plt.plot(x, y, **kwargs)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()
