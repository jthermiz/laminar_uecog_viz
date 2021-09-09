#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:35:48 2021

@author: vanessagutierrez
"""
from process_nwb.wavelet_transform import wavelet_transform

def zscore_data(trials_mat, num_base_pts=200):
    """Compute zscore across trial matrix
    Args:
        trials_matrix (nparray): Trial matrix of samples x trials
        num_base_pts (int, optional): The first num_base_pts are used for baseline. Defaults to 200.
    Returns:
            tm_norm_data (nparray): Normalized trial matrix
    """
    # Zscore the data
    mean = trials_mat[:num_base_pts].mean(axis=0, keepdims=True)
    std = trials_mat[:num_base_pts].std(axis=0, keepdims=True)
    tm_norm_data = (trials_mat - mean) / std    
    return tm_norm_data

def compute_high_gamma(trials_mat, fs, pre_buf=10000, post_buf=10000, baseline=200):
    """Computes high gamma using wavelet transform
    Args:
        trials_mat (list): List of np-arrays that contain trial matrices (samples x trials)
        fs (numeric): Sample rate
        pre_buf (int, optional): The number of samples to include prior to sample midpoint of trials_mat. Defaults to 10000.
        post_buf (int, optional): The number of samples to include after the sample midpoint. Defaults to 10000.
        baseline (int, optiona): The first set of samples to include in baseline. Defaults to 200.
    Returns:
            high_gamma: Returns zscored wavelet magnitude coefficients at high gamma frequencies, samples x trials
    """
    Xh, _, f, _  = wavelet_transform(trials_mat, rate=fs, filters='rat', hg_only=True)
    n = Xh.shape[0] // 2
    Xh = Xh[(n - pre_buf):(n + post_buf), :, :] #throw away edges bc of possible effect affects
    Xm = abs(Xh) #take abs value
    Xnorm = zscore_data(Xm, baseline) #zscore 
    high_gamma = Xnorm.mean(axis = -1)
    return high_gamma
