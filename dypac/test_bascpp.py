import numpy as np
import bascpp as bpp


def test_propagate_part():

    # examples of batch-level and consensus-level partitions
    # The following two batches are integer partitions on time windows
    part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
    part_cons = np.array([2, 1, 0, 2, 0, 1])

    # indices defining the batch
    n_batch = 2
    index_cons = [0, 3, 6]

    # Manually derived solution
    part_ground_truth = np.array([2, 1, 1, 0, 0, 2, 2, 2, 0, 1])

    part = bpp._propagate_part(part_batch, part_cons, n_batch, index_cons)
    assert np.min(part == part_ground_truth)
