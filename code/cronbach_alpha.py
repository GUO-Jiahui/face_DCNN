#!/usr/bin/env python3

import numpy as np
import read_data as rd

"""
Dec-16-2021 by Guo Jiahui and Ma Feilong

This script contains functions and examples to compute noise ceilings (Cronbach's alphas).

Run this script directly to see the example output.
"""


def cronbach_alpha(X):
    n_items = X.shape[0]
    alpha = (n_items / (n_items - 1.)) * (1. - X.var(axis=1).sum() / X.sum(axis=0).var())
    return alpha


if __name__ == '__main__':
    ### Below shows the example code to calculate the noise ceiling (Cronbash's alpha)
    ### for a neural ROI and for the behavioral arragement task.
    ### The neural noise ceilings contain both a run-wise and an all-stimuli version,
    ### and the behavioral noise ceiling only contains a run-wise version.
    ### For the run-wise version, the noise ceiling was calculated in each run, and averaged across runs.

    data_dir = '../data'

    ## roi: 'raFFA', 'laFFA'
    roi = 'raFFA'

    calphas_neural_rdms_run = []
    for run in range(12):
        neural_rdms_run = rd.get_neural_rdm(f'{data_dir}/neural_RDMs', roi, run=run)
        calphas_neural_rdms_run = cronbach_alpha(neural_rdms_run)
    calpha_mean_neural_rdms_run = np.mean(calphas_neural_rdms_run)
    print(f'--- CAlpha (mean of runs): neural ROI {roi} ---\n  calpha = {calpha_mean_neural_rdms_run}')

    calphas_behavioral_rdms_run = []
    for run in range(12):
        behavioral_rdms_run = rd.get_behavioral_rdm(f'{data_dir}/behavioral_RDMs', run=run)
        calphas_behavioral_rdms_run = cronbach_alpha(behavioral_rdms_run)
    calpha_mean_behavioral_rdms_run = np.mean(calphas_behavioral_rdms_run)
    print(f'--- CAlpha (mean of runs): behavioral task ---\n  calpha = {calpha_mean_behavioral_rdms_run}')

    ### all-stimuli version.
    neural_rdms = rd.get_neural_rdm(f'{data_dir}/neural_RDMs', roi, run='all')
    calpha_neural_rdms = cronbach_alpha(neural_rdms)
    print(f'--- CAlpha (all stimuli): neural ROI {roi} ---\n  calpha = {calpha_neural_rdms}')
