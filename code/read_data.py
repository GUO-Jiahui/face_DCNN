#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import squareform

"""
Dec-16-2021 by Guo Jiahui and Ma Feilong

This script contains functions to read RDMs and masks.

All RDMs are in the vectorized upper triangle format, i.e., the output of `squareform`.
For example, the vectorized upper triangle of the 707x707 RDM is a 1-D array with 249571 elements (707x706/2). 
Each single run has 59 or 58 elements, and the vectorized upper triangles have 1711 and 1653 elements, respectively.

The function `get_neural_rdm` loads neural RDM(s) of an ROI, i.e., RDMs based on fMRI responses.
The function `get_dcnn_rdm` loads a DCNN RDM of a DCNN layer.
The function `get_behavioral_rdm` loads behavioral RDM(s), i.e., RDMs based on the arrangement task.
The funciton `get_feature_mask` loads feature masks, which have the same shape as RDMs.
Each element is True if and only if the feature (e.g., perceived gender) is the same for a pair of stimuli.

When applicable, the function has a `mean` parameter.
When it's set to `True`, it returns a single RDM, which is the average across subjects and is used for RSA.
When it's set to `False`, it returns multiple RDMs, which can be used to compute Cronbach's alpha.
"""


def get_single_run_rdm(full_rdm, run=0):
    """
    Get the RDM of a single run based on the full 707x707 RDM.

    Parameters
    ----------
    full_rdm : ndarray of shape (249571, )
        The vectorized upper triangle of the 707x707 RDM.
    run : int
        The index of the run, one of {0, 1, ..., 11}.

    Returns
    -------
    rdm : ndarray of shape (1711, ) or (1653, )
        The vectorized upper triangle of the RDM for a single run.
    """
    full_mat = squareform(full_rdm)
    slc = slice(run*59, (run+1)*59)
    mat = full_mat[slc][:, slc]
    rdm = squareform(mat)
    return rdm


def get_neural_rdm(neural_rdm_dir, roi, mean=False, run='all'):
    """
    Parameters
    ----------
    neural_rdm_dir : str
        The directory containing neural RDMs.
    roi : str
        The name of the ROI, e.g., 'raFFA'.
    mean : bool
        Whether to average across all subjects.
    run : {'all', int}
        If it's 'all', returns vectorized 707x707 RDM(s).
        Otherwise, returns RDM(s) of a single run.

    Returns
    -------
    rdms : ndarray
        When `mean` is True, returns a single RDM which is the average across subjects.
        Otherwise, returns a 2-D array of stacked RDMs, where the length of the 1st dimension is the number of subjects.
        Depending on whether `run` is 'all' or an integer, the size of an RDM varies.
    """
    rdms = np.load(f'{neural_rdm_dir}/{roi}_RDMs.npy')
    if run != 'all':
        rdms = [get_single_run_rdm(_, run=run) for _ in rdms]
        rdms = np.stack(rdms, axis=0)
    if mean:
        rdms = rdms.mean(axis=0)
    return rdms


def get_dcnn_rdm(dcnn_rdm_dir, dcnn_name, layer_name, run='all'):
    """
    Parameters
    ----------
    dcnn_rdm_dir : str
        The directory containing DCNN RDMs.
    dcnn_name : str
        The name of the DCNN, e.g., 'Face_ArcFace'.
    layer_name : str
        The name of the DCNN layer, e.g., `fc1`.
    run : {'all', int}
        If it's 'all', returns vectorized 707x707 RDM(s).
        Otherwise, returns RDM(s) of a single run.
        
    Returns
    -------
    rdm : ndarray
        The RDM for the DCNN layer.
    """
    rdms = np.load(f'{dcnn_rdm_dir}/{dcnn_name}_RDMs.npz')
    rdm = rdms[layer_name]
    if run != 'all':
        rdm = get_single_run_rdm(rdm, run=run)
    return rdm


def get_behavioral_rdm(behavioral_rdm_dir, mean=False, run=0):
    """
    Parameters
    ----------
    behavioral_rdm_dir : str
        The directory containing behavioral RDMs.
    mean : bool
        Whether to average across all subjects.
    run : {'all', int}
        If it's 'all', returns vectorized 707x707 RDM(s).
        Otherwise, returns RDM(s) of a single run.

    Returns
    -------
    rdms : ndarray
        When `mean` is True, returns a single RDM which is the average across MTurkers.
        Otherwise, returns a 2-D array of stacked RDMs, where the length of the 1st dimension is the number of MTurkers.
    """
    rdms_run = np.load(f'{behavioral_rdm_dir}/{run:02d}.npy')
    if mean:
        rdm_run = np.mean(rdms_run, axis=0)
        return rdm_run
    return rdms_run


def get_feature_mask(mask_dir, feature_name, run):
    """
    Parameters
    ----------
    mask_dir : str
        The dictionary containing feature masks.
    feature_name : str
        The name of the feature, e.g., 'perceived_gender'.
    run : int
        The index of the run, one of {0, 1, ..., 11}.

    Returns
    -------
    mask : ndarray
        The `mask` is a boolean array which has the same shape as the corresponding RDM.
        It's `True` if and only if a pair of stimuli are in the same category for the feature.
    """
    masks = np.load(f'{mask_dir}/{feature_name}_masks.npy', allow_pickle=True)
    mask = masks[run]
    return mask
