#!/usr/bin/env python3

import numpy as np
from scipy.stats import pearsonr
import read_data as rd

"""
Dec-16-2021 by Guo Jiahui and Ma Feilong

This script contains functions and examples to run RSA.

Run this script directly to see the example output.
"""

DATA_DIR = '../data'


def run_rsa(vrdm1, vrdm2):
    [r, p] = pearsonr(vrdm1, vrdm2)
    return r


if __name__ == '__main__':
    ### Below shows the example code to run RSA between DCNN, neural, and behavioral RDMs.
    ### Other possbile parameters are available in the comment above each chosen one.
    roi = 'raFFA'  # roi: 'raFFA', 'laFFA'
    run = 0  # run: 0, 1, 2, ..., 11
    dcnn_idx = 0
    layer_idx = -1


    dcnn_names = ['Face_ArcFace', 'Face_AlexNet', 'Face_VGG16', 'Object_AlexNet', 'Object_VGG16']
    dcnn_name = dcnn_names[dcnn_idx]

    if dcnn_name == 'Face_ArcFace':
        layer_names = ['input'] + [f'_plus{i}' for i in range(49)] + ['pre_fc1'] + ['fc1']
    elif dcnn_name in ['Face_AlexNet', 'Object_AlexNet']:
        layer_names = ['input', 'conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc1', 'fc2']
    elif dcnn_name in ['Face_VGG16', 'Object_VGG16']:
        layer_names = ['input', 'block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool'] + ['block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool', 'fc1', 'fc2']
    layer_name = layer_names[layer_idx]

    neural_rdm = rd.get_neural_rdm(f'{DATA_DIR}/neural_RDMs', roi, mean=True)
    neural_rdm_run = rd.get_neural_rdm(f'{DATA_DIR}/neural_RDMs', roi, run=run, mean=True)
    dcnn_rdm_layer = rd.get_dcnn_rdm(f'{DATA_DIR}/DCNN_RDMs', dcnn_name, layer_name)
    dcnn_rdm_layer_run = rd.get_dcnn_rdm(f'{DATA_DIR}/DCNN_RDMs', dcnn_name, layer_name, run=run)
    behavioral_rdm_run = rd.get_behavioral_rdm(f'{DATA_DIR}/behavioral_RDMs', run=run, mean=True)

    r_dcnn_neural = run_rsa(dcnn_rdm_layer, neural_rdm)
    r_dcnn_neural_run = run_rsa(dcnn_rdm_layer_run, neural_rdm_run)
    r_dcnn_behavioral_run = run_rsa(dcnn_rdm_layer_run, behavioral_rdm_run)
    r_neural_behavioral_run = run_rsa(neural_rdm_run, behavioral_rdm_run)

    print(f'--- RSA: layer {layer_name} in {dcnn_name} & neural ROI {roi} ---\n  r = {r_dcnn_neural}')
    print(f'--- RSA of run number {run}: layer {layer_name} in {dcnn_name} & neural ROI {roi} ---\n  r = {r_dcnn_neural_run}')
    print(f'--- RSA of run number {run}: layer {layer_name} in {dcnn_name} & behavioral run number {run} ---\n  r = {r_dcnn_behavioral_run}')
    print(f'--- RSA of run number {run}: neural ROI {roi} & behavioral run number {run} ---\n  r = {r_neural_behavioral_run}')
