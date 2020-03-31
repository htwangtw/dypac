import numpy as np
import pytest
import dypac as dp


def _simu_tseries(n_time, n_roi, n_clusters, alpha):
    """ Simulate time series with a cluster structure for multiple ROIs.
        Returns:
            y (n_roi x n_time) the time series
            gt (n_roi) the ground truth partition
    """
    noise = np.random.normal(size=[n_roi , n_time]) # Some Gaussian random noise
    gt = np.zeros(shape=[n_roi,1]) # Ground truth clusters
    y = np.zeros(noise.shape) # The final time series
    ind = np.linspace(0,n_roi,n_clusters+1,dtype="int") # The indices for each cluster
    for cc in range(0, n_clusters): # for each cluster
        cluster = range(ind[cc], ind[cc + 1]) # regions for that particular cluster
        sig = np.random.normal(size=[1, n_time]) # a single signal
        y[cluster, :] = noise[cluster, :] + alpha * np.repeat(sig, ind[cc + 1] - ind[cc],0) # y = noise + a * signal
        gt[cluster] = cc # Adding the label for cluster in ground truth
    return y, gt


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

    part = dp._propagate_part(part_batch, part_cons, n_batch, index_cons)
    assert np.min(part == part_ground_truth)
