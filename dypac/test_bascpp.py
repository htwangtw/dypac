import numpy as np
import bascpp as bpp


def simu_tseries(n_time, n_roi, n_clusters, alpha):
    """Simulate time series with a cluster structure for multiple ROIs.
        Parameters
        ----------
        n_time: int
            number of simulated time points

        n_roi: int
            number of regions per cluster

        n_clusters: int
            number of clusters

        alpha: scalar, 0 <= . <= 1
            the std of the cluster signal (noise has a variance of 1)

        Returns
        -------
        y: ndarray size (n_roi * n_clusters) x n_time
            simulated time series
        gt: ndarray size n_roi
            the ground truth partition (cluster I is filled with Is)
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

def test_replicate_clusters():

    tseries, gt = simu_tseries(n_time=100, n_roi=30, n_clusters=3, alpha=2)
    n_replications = 10
    n_clusters = 3
    onehot = bpp.replicate_clusters(tseries, subsample_size=30, n_clusters=n_clusters, n_replications=n_replications, max_iter=30, n_init=10)

    # Check that onehot has the right dimensions
    assert onehot.shape == (n_replications * n_clusters, tseries.shape[0])
