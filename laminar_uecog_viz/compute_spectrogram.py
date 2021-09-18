#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:00:49 2021

@author: vanessagutierrez
"""

def compute_spectrogram(trials_mat, fs, pre_buf=1500, post_buf=1500, baseline=200):
        """Computes spectrogram using wavelet transform
            Args:
                trials_mat (list): List of np-arrays that contain trial matrices (samples x trials)
                fs (numeric): Sample rate
                pre_buf (int, optional): The number of samples to include prior to sample midpoint of trials_mat. Defaults to 1500.
                post_buf (int, optional): The number of samples to include after the sample midpoint. Defaults to 1500.
                baseline (int, optiona): The first set of samples to include in baseline. Defaults to 200.
            Returns:
                Xnorm, f (nparray, nparray): Returns zscored wavelet magnitude coefficients and corresponding frequencies
            """
        Xh, _, f, _  = wavelet_transform(trials_mat, rate=fs, filters='rat', hg_only=False)
        n = Xh.shape[0] // 2
        Xh = Xh[(n - pre_buf):(n + post_buf), :, :] #throw away edges bc of possible effect affects
        Xh = Xh[:, :, :] #throw away low frequencies
        f = f[:]
        Xm = abs(Xh) #take abs value
        Xnorm = viz.zscore_data(Xm, baseline) #zscore 
        print('f[0]', f[0])
        print('f[-1]', f[-1])
        return Xnorm, f