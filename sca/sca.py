"""
Stable Cluster Aggregation
"""
from random import sample
import numpy as np
from sklearn.cluster import k_means


def _part2onehot(part, n_clusters=0):
    """ Convert a partition with integer clusters into a series of one-hot
        encoding vectors.
    """
    if n_clusters == 0:
        n_clusters = np.max(part)+1

    onehot = np.zeros([part.shape[0], n_clusters])
    for cc in range(0, n_clusters):
        onehot[:, cc] = part == cc
    return onehot


def _select_subsample(y, subsample_size, contiguous=False):
    """ Select a random subsample in a data array
    """
    n_samples = y.shape[0]
    subsample_size = np.min([subsample_size, n_samples])
    max_start = n_samples - subsample_size
    if contiguous:
        start = np.floor((max_start + 1) * np.random.rand(1))
        stop = (start + subsample_size)
        samp = y[np.arange(int(start), int(stop)), :]
    else:
        samp = np.array(sample(y.tolist(), int(subsample_size)))
    return samp


def _replicate_cluster(y, subsample_size, n_clusters, n_replications=40,
                       contiguous=False, max_iter=30):
    """ Replicate a clustering on random subsamples
    """
    onehot = np.zeros([y.shape[0], n_clusters, n_replications])
    for ss in range(0, n_replications):
        samp = _select_subsample(y, subsample_size, contiguous)
        cent, part, inert = k_means(samp, n_clusters=n_clusters, init="random",
                                    max_iter=max_iter)
        onehot[:, :, ss] = _part2onehot(part, n_clusters)
    return onehot
