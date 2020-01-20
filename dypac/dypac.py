"""
Dynamic Parcel Aggregation with Clustering (dypac)
"""

# Authors: Pierre Bellec, Amal Boukhdir
# License: BSD 3 clause
import glob
import itertools

import load_confounds

from tqdm import tqdm

from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
import numpy as np

from sklearn.cluster import k_means
from sklearn.utils import check_random_state

from nilearn import EXPAND_PATH_WILDCARDS
from nilearn._utils.compat import Memory, Parallel, delayed, _basestring
from nilearn._utils.niimg import _safe_get_data
from nilearn._utils.niimg_conversions import _resolve_globbing
from nilearn._utils.cache_mixin import CacheMixin, cache
from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from nilearn.decomposition.base import BaseDecomposition


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
    ind_r = np.reshape(part, n_el) + np.repeat(np.array(range(n_part)) * n_clusters, n_voxel)
    ind_c = np.repeat(np.reshape(range(n_voxel), [1, n_voxel]), n_part, axis=0).flatten()
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


def _replicate_clusters(
    y, subsample_size, n_clusters, n_replications, max_iter, n_init, n_jobs, verbose
):
    """ Replicate a clustering on random subsamples
    """
    part = np.zeros([n_replications, y.shape[0]], dtype="int")
    list_start = _start_window(y.shape[1], n_replications, subsample_size)
    range_replication = range(n_replications)
    if verbose:
        range_replication = tqdm(range_replication)

    for rr in range_replication:
        samp = _select_subsample(y, subsample_size, list_start[rr])
        cent, part[rr,:], inert = k_means(
            samp,
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=max_iter,
            n_init=n_init,
            n_jobs=n_jobs,
        )
    return _part2onehot(part, n_clusters)


def _find_states(onehot, n_init, n_clusters, n_jobs, max_iter, threshold_sim):
    """Find dynamic states based on the similarity of clusters over time"""
    cent, states, inert = k_means(
        onehot,
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=max_iter,
        n_init=n_init,
        n_jobs=n_jobs,
    )

    for ss in range(n_clusters):
        if np.any(states == ss):
            parcels = onehot[states == ss, :]
            n_parcels = np.sum(states == ss)
            ref_cluster = csr_matrix(parcels.mean(dtype="float", axis=0))
            ref_sim = np.zeros(n_parcels)
            for pp in range(n_parcels):
                ref_sim[pp] = ref_cluster[parcels[pp,:]].mean(dtype="float")
            tmp = states[states == ss]
            tmp[ref_sim < threshold_sim] = n_clusters
            states[states == ss] = tmp
    return states


def _stab_maps(onehot, states, n_replications, n_states, threshold_stab):
    """Generate stability maps associated with different states"""
    dwell_time = np.zeros([n_states, 1])
    val = np.array([])
    col_ind = np.array([])
    row_ind = np.array([])

    for ss in range(0, n_states):
        dwell_time[ss] = np.sum(states == ss) / n_replications
        if np.any(states == ss):
            stab_map = onehot[states == ss, :].mean(dtype='float', axis=0)
            mask = stab_map > threshold_stab
            col_ind = np.append(col_ind, np.repeat(ss, np.sum(mask)))
            row_ind = np.append(row_ind, np.nonzero(mask)[1])
            val = np.append(val, stab_map[mask])
    stab_maps = csr_matrix((val, (row_ind, col_ind)), shape=[onehot.shape[1], n_states])
    # stab_maps = stab_maps[:,dwell_time>(1/n_replications)]
    # dwell_time = dwell_time[dwell_time>(1/n_replications)]
    return stab_maps, dwell_time


class dypac(BaseDecomposition):
    """Perform Stable Dynamic Cluster Analysis.

    Parameters
    ----------
    n_clusters: int
        Number of clusters to extract per time window

    n_init: int
        Number of initializations for k-means

    subsample_size: int
        Number of time points in a subsample

    n_states: int
        Number of expected states per cluster

    n_replications: int
        Number of replications of cluster analysis in each fMRI run

    max_iter: int
        Max number of iterations for k-means

    threshold_sim: float (0 <= . <= 1)
        Minimal acceptable average dice in a state

    threshold_stab: float (0 <= . <= 1), optional
        Minimal stability in the stability maps

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

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

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
        n_init=30,
        n_init_aggregation=100,
        subsample_size=30,
        n_states=3,
        n_replications=40,
        max_iter=30,
        threshold_sim=0.3,
        threshold_stab=0.1,
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
        n_jobs=1,
        verbose=0,
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
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Those settings are specific to parcel aggregation
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.n_init_aggregation = n_init_aggregation
        self.subsample_size = subsample_size
        self.n_clusters = n_clusters
        self.n_states = n_states
        self.n_replications = n_replications
        self.max_iter = max_iter
        self.threshold_sim = threshold_sim
        self.threshold_stab = threshold_stab

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
        if self.verbose:
            print("[{0}] Finding parcellation states".format(self.__class__.__name__))
        states = _find_states(
            onehot,
            self.n_init_aggregation,
            self.n_states * self.n_clusters,
            self.n_jobs,
            self.max_iter,
            self.threshold_sim,
        )

        # Generate the stability maps
        if self.verbose:
            print(
                "[{0}] Generating state stability maps".format(self.__class__.__name__)
            )
        stab_maps, dwell_time = _stab_maps(
            onehot, states, self.n_replications, self.n_states * self.n_clusters, self.threshold_stab
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

        if confounds is not None:
            counfounds = load_confounds.load_confounds(confounds)
        else:
            confounds = itertools.repeat(confounds)

        data_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._mask_and_cluster_single)(img=img, confound=confound)
            for img, confound in zip(imgs, confounds)
        )
        n_voxels = int(np.sum(_safe_get_data(self.masker_.mask_img_)))
        onehot = data_list[0]
        dwell_time = data_list[0][1]
        for i in range(1, len(data_list)):
            onehot = np.concat(onehot, data_list[i], axis=0)
            # Clear memory as fast as possible: remove the reference on
            # the corresponding block of data
            data_list[i] = None
        return onehot

    def _mask_and_cluster_single(self, img, confound):
        """Utility function for multiprocessing from _mask_and_reduce"""
        this_data = self.masker_.transform(img, confound)
        # Now get rid of the img as fast as possible, to free a
        # reference count on it, and possibly free the corresponding
        # data
        del img
        random_state = check_random_state(self.random_state)
        if self.verbose:
            print("[{0}] Replicating clustering".format(self.__class__.__name__))
        onehot = _replicate_clusters(
            this_data.transpose(),
            subsample_size=self.subsample_size,
            n_clusters=self.n_clusters,
            n_replications=self.n_replications,
            max_iter=self.max_iter,
            n_init=self.n_init,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        return onehot
