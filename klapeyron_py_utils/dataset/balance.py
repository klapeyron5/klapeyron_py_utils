import numpy as np


def get_N_data(datas, datas_weights):
    """
    :return: length of fused datas corresponding to their weights;
    according to the idiom: every single sample from the datas at least once with minimal possible resulting data length
    """
    assert len(datas) == len(datas_weights)
    assert sum(datas_weights) == 1
    assert all([x >= 0 for x in datas_weights])

    N_datas = 0
    for data, weight in zip(datas, datas_weights):
        if weight != 0:
            L = len(data)
            N = L / weight
        else:
            N = 0.0
        if N > N_datas:
            N_datas = N
    return N_datas


def one_lvl_datas_weights(datas, datas_weights):
    """
    :return: array from concatenating and shuffling every data from datas,
    with correspondence to their weights;
    the length of resulting array is possible minimum to provide every sample from every data from datas
    to appear at least once
    """
    N_datas = get_N_data(datas, datas_weights)
    N_datas = int(round(N_datas))
    balanced_datas = []
    for data, weight in zip(datas, datas_weights):
        N = int(round(N_datas * weight))
        assert N >= len(data)
        data = np.random.permutation(data)
        balanced_data = []
        while len(balanced_data) < N:
            balanced_data.extend(data)
        balanced_data = balanced_data[:N]
        balanced_datas.extend(balanced_data)
    assert abs(len(balanced_datas)-N_datas) <= len(datas)
    balanced_datas = np.random.permutation(balanced_datas)
    return balanced_datas


if __name__ == '__main__':
    def ut(datas, weights, N_gt=None):
        N_datas = get_N_data(datas, weights)
        if N_gt is not None:
            assert N_datas == N_gt
        balanced_data = one_lvl_datas_weights(datas, weights)
        if N_gt is not None:
            assert len(balanced_data) == N_gt
        return balanced_data


    datas = [
        [1, 2, 4],
        [5, 4, 3]
    ]
    weights = [0.5 , 0.5]
    N_gt = 6
    ut(datas, weights, N_gt)

    weights = [0.8, 0.5]
    try:
        ut(datas, weights, N_gt)
        assert False
    except Exception:
        pass

    datas = [
        [1, 2, 4, 7],
        [5, 4, 3]
    ]
    weights = [0.5, 0.5]
    N_gt = 8
    ut(datas, weights, N_gt)

    datas = [
        np.zeros((10,), int),
        np.ones((10,), int),
    ]
    weights = [0.5, 0.5]
    bd = ut(datas, weights, 20)
    assert set(bd) == {0, 1}
    un, counts = np.unique(bd, return_counts=True)

    datas = [
        np.zeros((20,), int),
        np.ones((10,), int),
    ]
    weights = [0.8, 0.2]
    bd = ut(datas, weights, 50)
    assert set(bd) == {0, 1}
    un, counts = np.unique(bd, return_counts=True)
    assert un[0] == 0
    assert un[1] == 1
    assert counts[0] == 40
    assert counts[1] == 10

    weights = [0.76, 0.24]
    bd = ut(datas, weights)
    assert set(bd) == {0, 1}
    un, counts = np.unique(bd, return_counts=True)
    assert un[0] == 0
    assert un[1] == 1
    assert counts[0] == 32
    assert counts[1] == 10

    weights = [0.77, 0.23]
    bd = ut(datas, weights)
    assert set(bd) == {0, 1}
    un, counts = np.unique(bd, return_counts=True)
    assert un[0] == 0
    assert un[1] == 1
    assert counts[0] == 33
    assert counts[1] == 10

    pass
