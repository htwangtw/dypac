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
