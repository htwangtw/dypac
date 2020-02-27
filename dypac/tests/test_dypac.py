import numpy as np
import dypac
import pytest
import pickle


def test_part2onehot():
    test = dict()
    test["part"] = np.array([[2, 1, 1], [3, 1, 1]])
    test["clusters"] = 5
    test["one_hot"] = [
        [False, False, False],
        [False, True, True],
        [True, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, True, True],
        [False, False, False],
        [True, False, False],
        [False, False, False],
    ]

    dypac_onehot = dypac._part2onehot(test["part"], test["clusters"]).toarray()
    assert np.array_equal(dypac_onehot, test["one_hot"])
    return 0


def test2_part2onehot():
    test = dict()
    test["part"] = np.array([[0]])
    test["clusters"] = 5
    test["one_hot"] = [[True], [False], [False], [False], [False]]

    dypac_onehot = dypac._part2onehot(test["part"], test["clusters"]).toarray()
    assert np.array_equal(dypac_onehot, test["one_hot"])


def test3_part2onehot():
    test = dict()
    test["part"] = np.array([[100]])
    test["clusters"] = 99

    with pytest.raises(ValueError, match="row index exceeds matrix dimensions"):
        dypac_onehot = dypac._part2onehot(test["part"], test["clusters"]).toarray()


def test4_part2onehot():
    test = dict()
    test["part"] = np.array([[]])
    test["clusters"] = 1
    test["one_hot"] = [[]]

    dypac_onehot = dypac._part2onehot(test["part"], test["clusters"]).toarray()
    assert np.array_equal(dypac_onehot, test["one_hot"])


def test5_part2onehot():
    test = dict()
    test["part"] = np.array([[-1]])
    test["clusters"] = 1

    with pytest.raises(ValueError, match="negative row index found"):
        dypac_onehot = dypac._part2onehot(test["part"], test["clusters"]).toarray()


def test6_part2onehot():
    test = dict()
    test["part"] = np.array([1])
    test["clusters"] = 10

    with pytest.raises(ValueError, match="expected 2, got 1"):
        dypac_onehot = dypac._part2onehot(test["part"], test["clusters"]).toarray()


def test7_part2onehot():
    test = dict()
    test["part"] = np.array([[True, False]])
    test["clusters"] = 5

    test["one_hot"] = [
        [False, True],
        [True, False],
        [False, False],
        [False, False],
        [False, False],
    ]

    dypac_onehot = dypac._part2onehot(test["part"], test["clusters"]).toarray()

    assert np.array_equal(dypac_onehot, test["one_hot"])


def test_start_window():

    n_time = 140
    n_replications = 5
    subsample_size = 30

    starts = [0, 27, 55, 83, 111]

    dypac_window = dypac._start_window(n_time, n_replications, subsample_size)

    assert np.array_equal(dypac_window, starts)


def test1_start_window():

    n_time = 140
    n_replications = 10
    subsample_size = 30

    starts = [0.0, 12.0, 24.0, 37.0, 49.0, 61.0, 74.0, 86.0, 98.0, 111.0]

    dypac_window = dypac._start_window(n_time, n_replications, subsample_size)

    assert np.array_equal(dypac_window, starts)


def test2_start_window():

    n_time = 29
    n_replications = 10
    subsample_size = 30

    starts = []

    dypac_window = dypac._start_window(n_time, n_replications, subsample_size)

    print(dypac_window)

    assert np.array_equal(dypac_window, starts)


def test3_start_window():

    n_time = 5
    n_replications = 10
    subsample_size = 30

    starts = []

    with pytest.raises(ValueError, match="-24, must be non-negative."):
        dypac_window = dypac._start_window(n_time, n_replications, subsample_size)


def test4_start_window():

    n_time = -1
    n_replications = 10
    subsample_size = 30

    starts = []

    with pytest.raises(ValueError, match="-30, must be non-negative."):
        dypac_window = dypac._start_window(n_time, n_replications, subsample_size)

    # return dypac_window


def test5_start_window():

    n_time = 140
    n_replications = 10
    subsample_size = 1

    starts = [0.0, 15.0, 31.0, 46.0, 62.0, 77.0, 93.0, 108.0, 124.0, 140.0]

    dypac_window = dypac._start_window(n_time, n_replications, subsample_size)

    assert np.array_equal(dypac_window, starts)


def test6_start_window():

    n_time = 140
    n_replications = 0
    subsample_size = 1

    starts = []
    dypac_window = dypac._start_window(n_time, n_replications, subsample_size)

    print(dypac_window)

    assert np.array_equal(dypac_window, starts)


def test7_start_window():

    n_time = 140
    n_replications = 120
    subsample_size = 130

    starts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    dypac_window = dypac._start_window(n_time, n_replications, subsample_size)

    print(dypac_window)

    assert np.array_equal(dypac_window, starts)


def test8_start_window():

    n_time = 0
    n_replications = 0
    subsample_size = 0

    starts = []
    dypac_window = dypac._start_window(n_time, n_replications, subsample_size)

    assert np.array_equal(dypac_window, starts)
