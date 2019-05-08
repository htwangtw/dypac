"""
Stable Cluster Aggregation
"""
import numpy as np
from sklearn.cluster import k_means


def _part2onehot(part, n_clusters=0):
    """ Convert a partition with integer clusters into a series of one-hot
        encoding vectors.
    """
    if n_clusters == 0:
        n_clusters = np.max(part)+1

    onehot = np.zeros([n_clusters, part.shape[0]])
    for cc in range(0, n_clusters):
        onehot[cc, :] = part == cc
    return onehot


def _select_subsample(y, subsample_size, contiguous=False):
    """ Select a random subsample in a data array
    """
    n_samples = y.shape[1]
    subsample_size = np.min([subsample_size, n_samples])
    max_start = n_samples - subsample_size
    if contiguous:
        start = np.floor((max_start + 1) * np.random.rand(1))
        stop = (start + subsample_size)
        samp = y[:, np.arange(int(start), int(stop))]
    else:
        samp = y[:, np.random.choice(n_samples, subsample_size)]
    return samp


def _replicate_cluster(y, subsample_size, n_clusters, n_replications=40,
                       contiguous=False, max_iter=30):
    """ Replicate a clustering on random subsamples
    """
    onehot = np.zeros([n_replications, n_clusters, y.shape[0]])
    for rr in range(0, n_replications):
        samp = _select_subsample(y, subsample_size, contiguous)
        cent, part, inert = k_means(samp, n_clusters=n_clusters, init="random",
                                    max_iter=max_iter)
        onehot[rr, :, :] = _part2onehot(part, n_clusters)
    return onehot


def recursive_cluster(y, subsample_size, n_clusters, n_states,
                      n_replications=40, contiguous=False, max_iter=30):
    """ Recursive k-means clustering of clusters based on random subsamples
    """
    onehot = _replicate_cluster(y, subsample_size, n_clusters, n_replications,
                                contiguous, max_iter)
    onehot = np.reshape(onehot, [n_replications * n_clusters, y.shape[0]])
    cent, part, inert = k_means(onehot, n_clusters=n_states, init="random",
                                max_iter=max_iter)
    stab_maps = np.zeros([y.shape[0], n_states])
    for ss in range(0, n_states):
        if np.any(part == ss):
            stab_maps[:, ss] = np.mean(onehot[part == ss, :], axis=0)
    return stab_maps
