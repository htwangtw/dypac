import numpy as np
import dypac
import pytest
import pickle

from scipy.sparse import csr_matrix, vstack, find


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

    assert np.array_equal(dypac_window, starts)


def test7_start_window():

    n_time = 140
    n_replications = 120
    subsample_size = 130

    starts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    dypac_window = dypac._start_window(n_time, n_replications, subsample_size)

    assert np.array_equal(dypac_window, starts)


def test8_start_window():

    n_time = 0
    n_replications = 0
    subsample_size = 0

    starts = []
    dypac_window = dypac._start_window(n_time, n_replications, subsample_size)

    assert np.array_equal(dypac_window, starts)


def test_select_subsample():

    y = np.zeros([4, 3])
    subsample_size = 1
    start = 1
    dypac_sample = dypac._select_subsample(y, subsample_size, start)
    subsample = [[0.0], [0.0], [0.0], [0.0]]
    assert np.array_equal(dypac_sample, subsample)


def test2_select_subsample():

    y = np.zeros([4, 3])
    subsample_size = 2
    start = 1
    dypac_sample = dypac._select_subsample(y, subsample_size, start)
    subsample = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    assert np.array_equal(dypac_sample, subsample)


def test3_select_subsample():

    y = np.zeros([4, 3])
    subsample_size = -2
    start = 1
    dypac_sample = dypac._select_subsample(y, subsample_size, start)
    subsample = np.empty((4, 0))
    assert np.array_equal(dypac_sample, subsample)


def test4_select_subsample():

    y = np.zeros([4, 3])
    subsample_size = 2
    start = -4
    subsample = np.empty((4, 0))

    with pytest.raises(IndexError, match="-4 is out of bounds for axis 1"):
        dypac_sample = dypac._select_subsample(y, subsample_size, start)


def test5_select_subsample():

    y = np.zeros([4, 3])
    subsample_size = 0
    start = 1
    dypac_sample = dypac._select_subsample(y, subsample_size, start)
    subsample = np.empty((4, 0))

    assert np.array_equal(dypac_sample, subsample)


def test6_select_subsample():

    y = np.zeros([4, 3])
    subsample_size = 1
    start = 0
    dypac_sample = dypac._select_subsample(y, subsample_size, start)
    subsample = [[0.0], [0.0], [0.0], [0.0]]

    assert np.array_equal(dypac_sample, subsample)


def test7_select_subsample():

    y = np.zeros([4, 3])
    subsample_size = 0
    start = 0
    dypac_sample = dypac._select_subsample(y, subsample_size, start)
    subsample = np.empty((4, 0))

    assert np.array_equal(dypac_sample, subsample)


def test8_select_subsample():

    y = np.zeros([0, 0])
    subsample_size = 1
    start = 1
    dypac_sample = dypac._select_subsample(y, subsample_size, start)
    subsample = np.empty((0, 0))

    assert np.array_equal(dypac_sample, subsample)


def test_stab_maps():
    val = [[0, 1], [4, 0], [5, 0]]
    states = np.array([1, 1, 2])
    n_replications = 2
    n_states = 2
    onehot = csr_matrix((val), dtype="bool")

    dypac_stab_maps = dypac._stab_maps(onehot, states, n_replications, n_states)[0]

    stab_maps = csr_matrix([[0.5, 0], [0.5, 0]], dtype="float64")

    assert np.array_equal(
        dypac_stab_maps.toarray(), stab_maps.toarray()
    ) and np.array_equal(dypac_stab_maps.indices, stab_maps.indices)


def test2_stab_maps():
    val = [[0, 1], [100, 0]]
    states = np.array([1, 1, 2])
    n_replications = 2
    n_states = 2
    onehot = csr_matrix((val), dtype="bool")

    dypac_stab_maps = dypac._stab_maps(onehot, states, n_replications, n_states)[0]

    stab_maps = csr_matrix([[0.5, 0], [0.5, 0]], dtype="float64")

    assert np.array_equal(
        dypac_stab_maps.toarray(), stab_maps.toarray()
    ) and np.array_equal(dypac_stab_maps.indices, stab_maps.indices)


def test3_stab_maps():
    val = [[0, 1], [100, 0], [-1, 0]]
    states = np.array([1, 2, 4])
    n_replications = 2
    n_states = 2
    onehot = csr_matrix((val), dtype="bool")

    dypac_stab_maps = dypac._stab_maps(onehot, states, n_replications, n_states)[0]
    stab_maps = csr_matrix([[0, 0], [1, 0]], dtype="float64")

    assert np.array_equal(
        dypac_stab_maps.toarray(), stab_maps.toarray()
    ) and np.array_equal(dypac_stab_maps.indices, stab_maps.indices)


def test4_stab_maps():
    val = [[0, 1], [5, 0]]
    states = np.array([])
    n_replications = 2
    n_states = 2
    onehot = csr_matrix((val), dtype="bool")

    dypac_stab_maps = dypac._stab_maps(onehot, states, n_replications, n_states)[0]
    stab_maps = csr_matrix([[0, 0], [0, 0]], dtype="float64")

    assert np.array_equal(
        dypac_stab_maps.toarray(), stab_maps.toarray()
    ) and np.array_equal(dypac_stab_maps.indices, stab_maps.indices)


def test5_stab_maps():
    val = [[0, 1], [5, 0]]
    states = np.array([1, 2, 7])
    n_replications = -1
    n_states = 2
    onehot = csr_matrix((val), dtype="bool")

    dypac_stab_maps = dypac._stab_maps(onehot, states, n_replications, n_states)[0]
    stab_maps = csr_matrix([[0, 0], [0, 1]], dtype="float64")

    assert np.array_equal(
        dypac_stab_maps.toarray(), stab_maps.toarray()
    ) and np.array_equal(dypac_stab_maps.indices, stab_maps.indices)


def test6_stab_maps():
    val = [[0, 1], [5, 0]]
    states = np.array([1, 2, 7])
    n_replications = 2
    n_states = -1
    onehot = csr_matrix((val), dtype="bool")

    with pytest.raises(ValueError, match="negative dimensions are not allowed"):
        dypac_stab_maps = dypac._stab_maps(onehot, states, n_replications, n_states)[0]


def test7_stab_maps():
    val = [[0, 1, 0, 9, 3,], [5, 0, 8, 3, 1], [4, 11, 2, 4, 5]]
    states = np.array([1, 2, 10])
    n_replications = 2
    n_states = 2
    onehot = csr_matrix((val), dtype="bool")

    stab_maps = csr_matrix([[0, 0], [1, 0], [0, 0], [1, 0], [1, 0]], dtype="float64")

    # with pytest.raises(ValueError, match="negative dimensions are not allowed"):
    dypac_stab_maps = dypac._stab_maps(onehot, states, n_replications, n_states)[0]

    assert np.array_equal(
        dypac_stab_maps.toarray(), stab_maps.toarray()
    ) and np.array_equal(dypac_stab_maps.indices, stab_maps.indices)


def test8_stab_maps():
    val = [[]]
    states = np.array([1, 2, 10])
    n_replications = 2
    n_states = 2
    onehot = csr_matrix((val), dtype="bool")

    dypac_stab_maps = dypac._stab_maps(onehot, states, n_replications, n_states)[0]

    if dypac_stab_maps.toarray().size == 0:
        assert True


def test_propagate_part():

    n_batch = 2
    part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1])
    index_batch = [0, 5, 10]
    index_cons = [0, 3, 6]
    dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)
    part = np.array([2, 0, 1, 1, 0, 2, 2, 2, 1, 1])

    assert np.array_equal(part, dypac_part)


def test2_propagate_part():

    n_batch = 2
    part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1])
    index_batch = [0, 5, 10]
    index_cons = [0, 3, 6]
    dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)
    part = np.array([2, 0, 1, 1, 0, 2, 2, 2, 1, 1])

    assert np.array_equal(part, dypac_part)


def test3_propagate_part():

    n_batch = 1
    part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1])
    index_batch = [0, 5, 10]
    index_cons = [0, 3, 6]
    dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)
    part = np.array([2, 1, 1, 0, 0, 2, 2, 2, 1, 0])

    print(np.array_equal(part, dypac_part))


def test4_propagate_part():

    n_batch = 3
    part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1])
    index_batch = [0, 5, 10]
    index_cons = [0, 3, 6]

    with pytest.raises(IndexError, match="list index out of range"):
        dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)


def test5_propagate_part():

    n_batch = -1
    part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1])
    index_batch = [0, 5, 10]
    index_cons = [0, 3, 6]
    dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)

    part = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    assert np.array_equal(part, dypac_part)


def test6_propagate_part():

    n_batch = 0
    part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1])
    index_batch = [0, 5, 10]
    index_cons = [0, 3, 6]
    dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)

    part = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    assert np.array_equal(part, dypac_part)


def test7_propagate_part():

    n_batch = 2
    part_batch = np.array([-1, 1, 1, 2, 2, 0, 0, 2, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1])
    index_batch = [0, 5, 10]
    index_cons = [0, 3, 6]
    dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)

    part = np.array([0, 0, 1, 1, 0, 2, 2, 1, 1, 1])

    assert np.array_equal(part, dypac_part)


def test8_propagate_part():

    n_batch = 2
    part_batch = np.array([0])
    part_cons = np.array([2, 1, 0, 2, 0, 1])
    index_batch = [0, 5, 10]
    index_cons = [0, 3, 6]
    dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)

    part = np.array([2])
    assert np.array_equal(part, dypac_part)


def test9_propagate_part():

    n_batch = 2
    part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1, 3, 1, 2, 8])
    index_batch = [0, 5, 10]
    index_cons = [0, 3, 6]
    dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)
    part = np.array([2, 0, 1, 1, 0, 2, 2, 2, 1, 1])
    assert np.array_equal(part, dypac_part)


def test10_propagate_part():
    n_batch = 2
    part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1])
    index_batch = []
    index_cons = [0, 3, 6]
    dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)
    part = np.array([2, 0, 1, 1, 0, 2, 2, 2, 1, 1])

    assert np.array_equal(part, dypac_part)


def test11_propagate_part():
    n_batch = 2
    part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1])
    index_batch = [0, 5, 10]
    index_cons = []

    with pytest.raises(IndexError, match="list index out of range"):
        dypac_part = dypac._propagate_part(part_batch, part_cons, n_batch, index_cons)
