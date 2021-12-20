#!/usr/bin/env python3

import numpy as np
from scipy.stats import zscore
import read_data as rd

"""
Dec-16-2021 by Guo Jiahui and Ma Feilong

This script contains functions and examples to calculate the difference between between-group distances and within-group distances for facial feature groups based on z-scored RDMs.
A larger difference of between-group vs. within-group distances indicates a clearer division between feature groups (e.g., female/male) in the representational space.
This script runs an example analysis for the results shown in Figure 6.

Run this script directly to see the example output.
"""

DATA_DIR = '../data'


def calc_diff_run(rdm_run_orig, mask_run):
    rdm_run = zscore(rdm_run_orig)
    within = np.mean(rdm_run[mask_run==1])
    between = np.mean(rdm_run[mask_run==0])
    diff = between - within
    return diff


if __name__ == '__main__':
    """
    Below shows the example code to calculate difference in the between- and within-group distance
    of an example face feature in the z-scored representational geometries
    of the behavioral arrangement task, of an example layer in a DCNN, and of a neural ROI.
    Other possbile parameters are available in the comment above each chosen one.
    Difference was calculated in each run, and averaged across runs.
    """

    roi = 'raFFA'  # roi: 'raFFA', 'laFFA'
    dcnn_idx = 0
    layer_idx = -1
    feature_idx = 0

    dcnn_names = ['Face_ArcFace', 'Face_AlexNet', 'Face_VGG16', 'Object_AlexNet', 'Object_VGG16']
    dcnn_name = dcnn_names[dcnn_idx]

    if dcnn_name == 'Face_ArcFace':
        layer_names = ['input'] + [f'_plus{i}' for i in range(49)] + ['pre_fc1'] + ['fc1']
    elif dcnn_name in ['Face_AlexNet', 'Object_AlexNet']:
        layer_names = ['input', 'conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc1', 'fc2']
    elif dcnn_name in ['Face_VGG16', 'Object_VGG16']:
        layer_names = ['input', 'block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool'] + ['block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool', 'fc1', 'fc2']
    layer_name = layer_names[layer_idx]

    feature_names = ['age', 'ethnicity', 'expression', 'head_orientation', 'perceived_gender']
    feature_name = feature_names[feature_idx]

    diffs_neural = []
    diffs_dcnn_layer = []
    diffs_behavioral = []
    for run in range(12):
        neural_rdm_run = rd.get_neural_rdm(f'{DATA_DIR}/neural_RDMs', roi, run=run, mean=True)
        dcnn_rdm_layer_run = rd.get_dcnn_rdm(f'{DATA_DIR}/DCNN_RDMs', dcnn_name, layer_name, run=run)
        behavioral_rdm_run = rd.get_behavioral_rdm(f'{DATA_DIR}/behavioral_RDMs', run=run, mean=True)
        mask_run = rd.get_feature_mask(f'{DATA_DIR}/feature_masks', feature_name, run=run)
        diffs_neural.append(calc_diff_run(neural_rdm_run, mask_run))
        diffs_dcnn_layer.append(calc_diff_run(dcnn_rdm_layer_run, mask_run))
        diffs_behavioral.append(calc_diff_run(behavioral_rdm_run, mask_run))

    diff_neural_mean = np.mean(diffs_neural)
    diff_dcnn_layer_mean = np.mean(diffs_dcnn_layer)
    diff_behavioral_mean = np.mean(diffs_behavioral)

    print(f'--- Diff (between-within) in feature {feature_name}: neural ROI {roi} ---\n  diff = {diff_neural_mean}')
    print(f'--- Diff (between-within) in feature {feature_name}: layer {layer_name} in {dcnn_name} ---\n  diff = {diff_dcnn_layer_mean}')
    print(f'--- Diff (between-within) in feature {feature_name}: behavioral task ---\n  diff = {diff_behavioral_mean}')
