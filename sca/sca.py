"""
Stable Cluster Aggregation
"""
import numpy as np
from sklearn.cluster import k_means
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

class CommunityState():

    def __init__(self, subsample_size=30, n_clusters=10,  n_states=3,
                 n_replications=40, contiguous=False, max_iter=30,
                 threshold_dice=0.3, threshold_stability=0.5, n_init=10,
                 n_jobs=1, n_init_aggregation=100):
        self.subsample_size = subsample_size
        self.n_clusters = n_clusters
        self.n_states = n_states
        self.n_replications = n_replications
        self.contiguous = contiguous
        self.max_iter = max_iter
        self.threshold_dice = threshold_dice
        self.threshold_stability = threshold_stability
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.n_init_aggregation = n_init_aggregation

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
                       contiguous=False, max_iter=30, n_init=10, n_jobs=1):
    """ Replicate a clustering on random subsamples
    """
    onehot = np.zeros([n_replications, n_clusters, y.shape[0]], dtype='bool')
    for rr in range(0, n_replications):
        samp = _select_subsample(y, subsample_size, contiguous)
        cent, part, inert = k_means(samp, n_clusters=n_clusters,
                                    init="k-means++", max_iter=max_iter,
                                    n_init=n_init, n_jobs=n_jobs)
        onehot[rr, :, :] = _part2onehot(part, n_clusters)
    return onehot


def _dice(vec1, vec2):
    """ Dice between two binary vectors
    """
    d = 2 * np.sum((vec1 == 1) & (vec2 == 1)) / (np.sum(vec1) + np.sum(vec2))
    return d


def _dice_vec(vec, onehot):
    size_vec = np.sum(vec)
    size_matrix = np.sum(onehot, axis=1)
    dice = np.matmul(onehot, vec.transpose())
    dice = 2 * np.divide(dice, (size_vec + size_matrix))
    return dice

def _dice_matrix(onehot):
    n_part = onehot.shape[0]
    dice_matrix = np.zeros((n_part, n_part))
    for pp1 in range(n_part-1):
        for pp2 in range(pp1+1,n_part):
            dice_matrix[pp1, pp2] = _dice(onehot[pp1, :], onehot[pp2, :])
    dice_matrix = dice_matrix + np.transpose(dice_matrix)
    idx = np.diag_indices(dice_matrix.shape[0])
    dice_matrix[idx] = 1
    return dice_matrix

def recursive_cluster(y, subsample_size, n_clusters,  n_states,
                      n_replications=40, contiguous=False, max_iter=30,
                      threshold_dice=0.3, threshold_stability=0.5,
                      n_init=10, n_jobs=1, n_init_aggregation=100):
    """ Recursive k-means clustering of clusters based on random subsamples
    """
    onehot = _replicate_cluster(y, subsample_size, n_clusters, n_replications,
                                contiguous, max_iter, n_init, n_jobs)
    onehot = np.reshape(onehot, [n_replications * n_clusters, y.shape[0]])
    part = _kmeans_aggregation(onehot, n_init_aggregation,
                               n_states * n_clusters, n_jobs, max_iter,
                               threshold_stability, threshold_dice)
    print(part)
    stab_maps, dwell_time = _stab_maps(onehot, part, n_replications,
                                       n_states * n_clusters)
    return stab_maps, dwell_time


def _kmeans_aggregation(onehot, n_init, n_clusters, n_jobs, max_iter,
                        threshold_stability, threshold_dice):
    cent, part, inert = k_means(onehot, n_clusters=n_clusters,
                                init="k-means++", max_iter=max_iter,
                                n_init=n_init, n_jobs=n_jobs)
    for ss in range(n_clusters):
        if np.any(part == ss):
            ref_cluster = np.mean(onehot[part == ss, :], axis=0)
            ref_cluster = ref_cluster > threshold_stability
            dice = _dice_vec(ref_cluster, onehot[part==ss,:])
            tmp = part[part==ss]
            tmp[dice<threshold_dice] = -1
            part[part==ss] = tmp
    return part


def _stab_maps(onehot, part, n_replications, n_clusters):
    stab_maps = np.zeros([onehot.shape[1], n_clusters])
    dwell_time = np.zeros([n_clusters, 1])
    for ss in range(0, n_clusters):
        dwell_time[ss] = np.sum(part==ss) / n_replications
        if np.any(part == ss):
            stab_maps[:, ss] = np.mean(onehot[part == ss, :], axis=0)
    return stab_maps, dwell_time


def _hierarchical_aggregation(onehot):
    dmtx = _dice_matrix(onehot)
    iu = np.triu_indices(dmtx.shape[0], 1)
    dist_part = 1 - dmtx[iu]
    hier_clustering = linkage(dist_part, method="average",
                              optimal_ordering=True)
    states = fcluster(hier_clustering, 1-threshold_dice, criterion='distance')
    n_clusters = np.max(states)
    stab_maps = np.zeros([y.shape[0], n_states])
    dwell_time = np.zeros([n_states, 1])
    for ss in range(0, n_states):
        dwell_time[ss] = np.sum(states==ss) / n_replications
        if np.any(states == ss):
            stab_maps[:, ss] = np.mean(onehot[states == ss, :], axis=0)
    return stab_maps, dwell_time
