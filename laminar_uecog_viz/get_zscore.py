#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:52:45 2021

@author: vanessagutierrez
"""

import numpy as np


def stimulus_onset_one_channel(trials_dict, channel, onset_start, onset_stop):
    """
    Parameters
    ----------
    trials_dict (dict): Trials dictionary that is the length of number of channels and each channel key has its trial matrix (samples, trials)
    channel (int): Specific channel you want data for.
    onset_start (int): starting period of the stimulus onset including some ms of baseline before.
    onset_stop (int): ending period of the stimulus onset including some ms of baseline after.

    Returns
    -------
    onset_for_channel (np.array): array with the data during the onset period for all trials.

    """
    
    all_data_for_channel = trials_dict[channel].T
    onset_for_channel = []
    onset_start = int(onset_start)
    onset_stop = int(onset_stop)
    
    for trial in range(len(all_data_for_channel)): 
        one_trial = all_data_for_channel[trial]
        onset_for_channel.append(one_trial[onset_start:onset_stop])
        
    return np.array(onset_for_channel)


def zscore_from_baseline(trials_dict, channel, onset_start, onset_stop):
    """
    

    Parameters
    ----------
    trials_dict (dict): Trials dictionary that is the length of number of channels and each channel key has its trial matrix (samples, trials)
    channel (int): Specific channel you want data for.
    onset_start (int): starting period of the stimulus onset including some ms of baseline before.
    onset_stop (int): ending period of the stimulus onset including some ms of baseline after.

    Returns
    -------
    zscored (array): zscored data for channel.

    """
    onset_data = stimulus_onset_one_channel(trials_dict, channel, onset_start, onset_stop)
    
    average_onset = np.average(onset_data)
    standard_dev = np.std(onset_data)
    
    all_data_for_channel = trials_dict[channel].T #returns data for one channel for all trials
    zscored = (all_data_for_channel - average_onset)/(standard_dev)
        
    return zscored    


def get_average_zscore(trials_dict, channel, onset_start, onset_stop):
    """
    

    Parameters
    ----------
    trials_dict (dict): Trials dictionary that is the length of number of channels and each channel key has its trial matrix (samples, trials)
    channel (int): Specific channel you want data for.
    onset_start (int): starting period of the stimulus onset including some ms of baseline before.
    onset_stop (int): ending period of the stimulus onset including some ms of baseline after.

    Returns
    -------
    mean_zscore (array): mean zscored trial for channel.

    """
 
    data_for_channel = zscore_from_baseline(trials_dict, channel, onset_start, onset_stop)
    
    different_format = np.transpose(data_for_channel) #array shape(12000, 60)

    
#     trial_mat = np.zeros((tmax, len(data_for_channel))) #array shape(12000, 60)

    
#     for timepoint in range(len(different_format)): # for each timepoint in all of 60 trials
#         sub_trial = different_format[timepoint] #this gives us the data for the first timepoint across 60 trials when timepoint = 0

#         trial_mat[timepoint, :] = sub_trial
    
    mean_zscore = np.mean(different_format, axis = 1)
    
    return mean_zscore


def zscore_data(trials_mat, num_base_pts=600):
    """
    Compute zscore across trial matrix
        created by Izzy
    Args:
        trials_matrix (nparray): Trial matrix of samples x trials of one channel
        num_base_pts (int, optional): The first num_base_pts are used for baseline. Defaults to 200.
    Returns:
            tm_norm_data (nparray): Normalized trial matrix

    Parameters
    ----------
    trials_mat (nparray): Trial matrix of samples x trials
    num_base_pts (int, optional): The first num_base_pts are used for baseline. Defaults to 600.

    Returns
    -------
    zscored (array): zscored data for channel.

    """
    
    # Zscore the data
    mean = trials_mat[:num_base_pts].mean(axis=0, keepdims=True)
    std = trials_mat[:num_base_pts].std(axis=0, keepdims=True)
    zscored = (trials_mat - mean) / std    
    
    return zscored


