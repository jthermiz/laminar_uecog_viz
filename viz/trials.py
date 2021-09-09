#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:09:56 2021

@author: vanessagutierrez
"""
import numpy as np 


def get_ch_trials_matrix(signal_data, marker_onsets, channel, pre_buf = 10000, post_buf = 10000):
    
    """Returns trial matrix
    Args:
        signal_data (np.array): signal data (nsamples, nchannels)
        markers (list): List of trial onset in samples
        pre_buf (int, optional): Number of samples to pull prior to baseline. Defaults to 10000.
        post_buf (int, optional): Number of samples to pull after. Defaults to 10000.
    Returns:
        trials_mat (np.array): Trial matrix for one channel (samples, trials)
    """
    
    nsamples = post_buf + pre_buf
    ntrials = len(marker_onsets)
    trials_mat = np.empty((nsamples, ntrials))
    channel_data = signal_data[:, channel]
    
    for idx, marker in enumerate(marker_onsets):
        start_frame, end_frame = marker - pre_buf, marker + post_buf
        trials_mat[:, idx] = channel_data[int(start_frame):int(end_frame)]
    return trials_mat

def get_all_trials_matrices(signal_data, marker_onsets, channel_order, pre_buf = 10000, post_buf = 10000):
    """
    Returns
    -------
    python dictionary where the key is the channel and the value is the trial matrix for that channel
    now, instead of calling get_trials_matrix a bunch of times when we want to visualize, we can 
    iterate over the keys in the all_trials matrix 

    """
    
    all_trials = {}
    for i in np.arange(len(channel_order)):
        one_channel = get_ch_trials_matrix(signal_data, marker_onsets, i)
        all_trials[channel_order[i]] = one_channel
    trials_dict = all_trials
    return trials_dict
