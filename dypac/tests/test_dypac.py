import numpy as np
import dypac
import pytest

test_one = dict()
test_one["part"] = np.array([[2, 1, 1], [3, 1, 1]])
test_one["clusters"] = 5
test_one["one_hot"] = [
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

test_two = dict()
test_two["part"] = np.array([[15, 7], [12, 17]])
test_two["clusters"] = 20
test_two["one_hot"] = [
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, True],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [True, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [True, False],
    [False, False],
    [False, False],
    [False, False],
    [False, False],
    [False, True],
    [False, False],
    [False, False],
]


test_three = dict()
test_three["part"] = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
test_three["clusters"] = 5
test_three["one_hot"] = [
    [False, False, False, False],
    [True, True, True, True],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [True, True, True, True],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [True, True, True, True],
    [False, False, False, False],
]


def test_part2onehot():

    dypac_onehot = dypac._part2onehot(test_one["part"], test_one["clusters"]).toarray()
    assert np.array_equal(dypac_onehot, test_one["one_hot"])

    dypac_onehot2 = dypac._part2onehot(test_two["part"], test_two["clusters"]).toarray()
    assert np.array_equal(dypac_onehot2, test_two["one_hot"])

    dypac_onehot3 = dypac._part2onehot(
        test_three["part"], test_three["clusters"]
    ).toarray()
    assert np.array_equal(dypac_onehot3, test_three["one_hot"])
