"""
Dynamic Parcel Aggregation with Clustering (dypac)
"""

# Authors: Pierre Bellec, Amal Boukhdir
# License: BSD 3 clause
import glob
import itertools

from tqdm import tqdm

from scipy.sparse import csr_matrix, vstack, find
import numpy as np

import ./bascpp as bpp
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

from nilearn import EXPAND_PATH_WILDCARDS
from nilearn._utils.niimg import _safe_get_data
from nilearn._utils.niimg_conversions import _resolve_globbing
from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from nilearn.decomposition.base import BaseDecomposition
from nilearn.image import new_img_like


def _select_subsample(y, subsample_size, start=None):
    """ Select a random subsample in a data array
    """
    n_samples = y.shape[1]
    subsample_size = np.min([subsample_size, n_samples])
    max_start = n_samples - subsample_size
    if start is not None:
        start = np.min([start, max_start])
    else:
        start = np.floor((max_start + 1) * np.random.rand(1))
    stop = start + subsample_size
    samp = y[:, np.arange(int(start), int(stop))]
    return samp


def _part2onehot(part, n_clusters=0):
    """ Convert a series of partition (one per row) with integer clusters into
        a series of one-hot encoding vectors (one per row and cluster).
    """
    if n_clusters == 0:
        n_clusters = np.max(part) + 1
    n_part, n_voxel = part.shape
    n_el = n_part * n_voxel
    val = np.repeat(True, n_el)
    ind_r = np.reshape(part, n_el) + np.repeat(
        np.array(range(n_part)) * n_clusters, n_voxel
    )
    ind_c = np.repeat(
        np.reshape(range(n_voxel), [1, n_voxel]), n_part, axis=0
    ).flatten()
    s_onehot = [n_part * n_clusters, n_voxel]
    onehot = csr_matrix((val, (ind_r, ind_c)), shape=s_onehot, dtype="bool")
    return onehot


def _start_window(n_time, n_replications, subsample_size):
    """ Get a list of the starting points of sliding windows.
    """
    max_replications = n_time - subsample_size + 1
    n_replications = np.min([max_replications, n_replications])
    list_start = np.linspace(0, max_replications, n_replications)
    list_start = np.floor(list_start)
    list_start = np.unique(list_start)
    return list_start


def _propagate_part(part_batch, part_cons, n_batch, index_cons):
    """ Combine partitions across batches with the within-batch partitions
    """
    part = np.zeros(part_batch.shape, dtype=np.int32)
    for bb in range(n_batch):
        range_batch = np.unique(np.floor(np.arange(bb, part_batch.shape[0], n_batch)).astype("int"))
        range_cons = range(index_cons[bb], index_cons[bb+1])
        sub_batch = part_batch[range_batch]
        sub_cons = part_cons[range_cons]
        part[range_batch] = sub_cons[sub_batch]

    return part


def _kmeans_batch( onehot, n_clusters, init="random", max_iter=30, n_batch=2, n_init=10, verbose=False, threshold_sim=0.3):
    """ Iteration of consensus clustering over batches of onehot
    """

    # Iterate across all batches
    part_batch = np.zeros([onehot.shape[0]], dtype="int")
    for bb in tqdm(range(n_batch), disable=not verbose, desc="Intra-batch consensus"):
        index_batch = np.unique(np.floor(np.arange(bb, onehot.shape[0], n_batch)).astype("int"))
        centroids, part, inert = k_means(
            onehot[index_batch, :],
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=max_iter,
            n_init=n_init,
        )
        if bb == 0:
            curr_pos = centroids.shape[0]
            index_cons = np.array([0, curr_pos])
            stab_batch = csr_matrix(centroids)
        else:
            curr_pos = curr_pos + centroids.shape[0]
            index_cons = np.hstack([index_cons, curr_pos])
            stab_batch = vstack([stab_batch, csr_matrix(centroids)])
        part_batch[index_batch] = part

    # Now apply consensus clustering on the binarized centroids
    cent, part_cons, inert = k_means(
            stab_batch,
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=max_iter,
            n_init=n_init,
        )

    # Finally propagate the batch level partition to the original onehots
    part = _propagate_part(part_batch, part_cons, n_batch, index_cons)

    return part


def _replicate_clusters(
    y, subsample_size, n_clusters, n_replications, max_iter, n_init, verbose, embedding=np.array([]), desc="", normalize=False
):
    """ Replicate a clustering on random subsamples
    """
    part = np.zeros([n_replications, y.shape[0]], dtype="int")
    list_start = _start_window(y.shape[1], n_replications, subsample_size)
    range_replication = range(n_replications)

    for rr in tqdm(range_replication, disable=not verbose, desc=desc):
        samp = _select_subsample(y, subsample_size, list_start[rr])
        if normalize:
            samp = scale(samp, axis=1)
        if embedding.shape[0] > 0:
            samp = np.concatenate([samp, embedding], axis=1)
        cent, part[rr, :], inert = k_means(
            samp,
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=max_iter,
            n_init=n_init,
        )
    return _part2onehot(part, n_clusters)


def _find_states(onehot, n_states=10, max_iter=30, threshold_sim=0.3, n_batch=0, n_init=10, verbose=False):
    """Find dynamic states based on the similarity of clusters over time"""
    if n_batch > 1:
        states = _kmeans_batch(
            onehot,
            n_clusters=n_states,
            init="random",
            max_iter=max_iter,
            threshold_sim=threshold_sim,
            n_batch=n_batch,
            n_init=n_init,
            verbose=verbose,
        )
    else:
        if verbose:
            print("Consensus clustering.")
        cent, states, inert = k_means(
            onehot,
            n_clusters=n_states,
            init="k-means++",
            max_iter=max_iter,
            n_init=n_init,
        )

    states = _trim_states(onehot, states, n_states, verbose, threshold_sim)
    return states


def _trim_states(onehot, states, n_states, verbose, threshold_sim):
    """Trim the states clusters to exclude outliers
    """
    for ss in tqdm(range(n_states), disable=not verbose, desc="Trimming states"):
        [ix, iy, val] = find(onehot[states == ss, :])
        size_onehot = np.array(onehot[states == ss, :].sum(axis=1)).flatten()
        ref_cluster = np.array(onehot[states == ss, :].mean(dtype="float", axis=0))
        avg_stab = np.bincount(ix, weights=ref_cluster[0,iy].flatten())
        avg_stab = np.divide(avg_stab, size_onehot)
        tmp = states[states == ss]
        tmp[avg_stab < threshold_sim] = n_states
        states[states == ss] = tmp
    return states

def _stab_maps(onehot, states, n_replications, n_states):
    """Generate stability maps associated with different states"""

    dwell_time = np.zeros(n_states)
    val = np.array([])
    col_ind = np.array([])
    row_ind = np.array([])

    for ss in range(0, n_states):
        dwell_time[ss] = np.sum(states == ss) / n_replications
        if np.any(states == ss):
            stab_map = onehot[states == ss, :].mean(dtype="float", axis=0)
            mask = stab_map > 0

            col_ind = np.append(col_ind, np.repeat(ss, np.sum(mask)))
            row_ind = np.append(row_ind, np.nonzero(mask)[1])
            val = np.append(val, stab_map[mask])
    stab_maps = csr_matrix((val, (row_ind, col_ind)), shape=[onehot.shape[1], n_states])

    # Re-order stab maps by descending dwell time
    indsort = np.argsort(-dwell_time)
    stab_maps = stab_maps[:, indsort]
    dwell_time = dwell_time[indsort]

    return stab_maps, dwell_time


class dypac(BaseDecomposition):
    """Perform Stable Dynamic Cluster Analysis.

    Parameters
    ----------
    n_clusters: int
        Number of clusters to extract per time window

    n_states: int
        Number of expected dynamic states

    n_replications: int
        Number of replications of cluster analysis in each fMRI run

    n_batch: int
        Number of batches to run through consensus clustering.
        If n_batch<=1, consensus clustering will be applied
        to all replications in one pass. Processing with batch will
        reduce dramatically the compute time, but will change slightly
        the results.

    n_init: int
        Number of initializations for k-means

    subsample_size: int
        Number of time points in a subsample

    max_iter: int
        Max number of iterations for k-means

    threshold_sim: float (0 <= . <= 1)
        Minimal acceptable average dice in a state

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    mask_strategy: {'background', 'epi' or 'template'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, 'epi' if they
        are raw EPI images, or you could use 'template' which will
        extract the gray matter part of your data by resampling the MNI152
        brain mask for your data's field of view.
        Depending on this value, the mask will be computed from
        masking.compute_background_mask, masking.compute_epi_mask or
        masking.compute_gray_matter_mask. Default is 'epi'.

    mask_args: dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    memory: instance of joblib.Memory or str
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    verbose: integer, optional
        Indicate the level of verbosity. By default, print progress.

    Attributes
    ----------
    `mask_img_` : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    """

    def __init__(
        self,
        n_clusters=10,
        n_states=3,
        n_replications=40,
        n_batch=1,
        n_init=30,
        n_init_aggregation=100,
        subsample_size=30,
        max_iter=30,
        threshold_sim=0.3,
        random_state=None,
        mask=None,
        smoothing_fwhm=None,
        standardize=True,
        detrend=True,
        low_pass=None,
        high_pass=None,
        t_r=None,
        target_affine=None,
        target_shape=None,
        mask_strategy="epi",
        mask_args=None,
        memory=Memory(cachedir=None),
        memory_level=0,
        verbose=1,
    ):
        # All those settings are taken from nilearn BaseDecomposition
        self.random_state = random_state
        self.mask = mask

        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self.memory = memory
        self.memory_level = max(0, memory_level + 1)
        self.verbose = verbose

        # Those settings are specific to parcel aggregation
        self.n_clusters = n_clusters
        self.n_states = n_states
        self.n_batch = n_batch
        self.n_replications = n_replications
        self.n_init = n_init
        self.n_init_aggregation = n_init_aggregation
        self.subsample_size = subsample_size
        self.max_iter = max_iter
        self.threshold_sim = threshold_sim

    def _check_components_(self):
        if not hasattr(self, "components_"):
            raise ValueError(
                "Object has no components_ attribute. "
                "This is probably because fit has not "
                "been called."
            )

    def fit(self, imgs, confounds=None):
        """Compute the mask and the dynamic parcels across datasets

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which the mask is calculated. If this is a list,
            the affine is considered the same for all.

        confounds: list of CSV file paths or 2D matrices
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details. Should match with the list
            of imgs given.

         Returns
         -------
         self: object
            Returns the instance itself. Contains attributes listed
            at the object level.

        """
        # Base fit for decomposition estimators : compute the embedded masker

        if isinstance(imgs, _basestring):
            if EXPAND_PATH_WILDCARDS and glob.has_magic(imgs):
                imgs = _resolve_globbing(imgs)

        if isinstance(imgs, _basestring) or not hasattr(imgs, "__iter__"):
            # these classes are meant for list of 4D images
            # (multi-subject), we want it to work also on a single
            # subject, so we hack it.
            imgs = [imgs]

        if len(imgs) == 0:
            # Common error that arises from a null glob. Capture
            # it early and raise a helpful message
            raise ValueError(
                "Need one or more Niimg-like objects as input, "
                "an empty list was given."
            )
        self.masker_ = check_embedded_nifti_masker(self)

        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit(imgs)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

        # mask_and_reduce step
        if self.verbose:
            print("[{0}] Loading data".format(self.__class__.__name__))
        onehot = self._mask_and_reduce(imgs, confounds)

        # find the states
        states = _find_states(
            onehot,
            n_states=self.n_states,
            max_iter=self.max_iter,
            threshold_sim=self.threshold_sim,
            n_batch=self.n_batch,
            n_init=self.n_init,
            verbose=self.verbose,
        )

        # Generate the stability maps
        stab_maps, dwell_time = _stab_maps(
            onehot, states, self.n_replications, self.n_states * self.n_clusters
        )

        # Return components
        self.components_ = stab_maps.transpose()
        self.dwell_time_ = dwell_time
        return self

    def _mask_and_reduce(self, imgs, confounds=None):
        """Uses cluster aggregation over sliding windows to estimate
        stable dynamic parcels from a list of 4D fMRI datasets.

        Returns
        ------
        stab_maps: ndarray or memorymap
            Concatenation of dynamic parcels across all datasets.
        """
        if not hasattr(imgs, "__iter__"):
            imgs = [imgs]

        if confounds is None:
            confounds = itertools.repeat(confounds)

        onehot = csr_matrix([0,])
        for ind, img, confound in zip(range(len(imgs)), imgs, confounds):
            if ind > 0:
                onehot = vstack(
                    [onehot, self._mask_and_cluster_single(img=img, confound=confound, ind=ind)]
                )
            else:
                onehot = self._mask_and_cluster_single(img=img, confound=confound, ind=ind)
        return onehot

    def _mask_and_cluster_single(self, img, confound, ind):
        """Utility function for _mask_and_reduce"""
        this_data = self.masker_.transform(img, confound)
        # Now get rid of the img as fast as possible, to free a
        # reference count on it, and possibly free the corresponding
        # data
        del img
        random_state = check_random_state(self.random_state)
        onehot = _replicate_clusters(
            this_data.transpose(),
            subsample_size=self.subsample_size,
            n_clusters=self.n_clusters,
            n_replications=self.n_replications,
            max_iter=self.max_iter,
            n_init=self.n_init,
            desc="Replicating clusters in data #{0}".format(ind),
            verbose=self.verbose,
        )
        return onehot

    def transform_sparse(self, img, confound=None):
        """Transform a 4D dataset in a component space"""
        self._check_components_()
        this_data = self.masker_.transform(img, confound)
        del img
        reg = LinearRegression().fit(
            self.components_.transpose(), this_data.transpose()
        )
        return reg.coef_

    def inverse_transform_sparse(self, weights):
        """Transform component weights as a 4D dataset"""
        self._check_components_()
        self.masker_.inverse_transform(weights * self.components_)
