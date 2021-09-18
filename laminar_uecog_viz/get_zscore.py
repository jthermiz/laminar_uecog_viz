#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:52:45 2021

@author: vanessagutierrez
"""

import numpy as np


def stimulus_onset_one_channel(data_for_trial_list, channel, onset_start, onset_stop):
    
    """inputs: 
        data for trial list(list)
            list with data with an array for each trial that contains subarrays for each channel
        channel(int)
             channel you want data for 
        onset_start (int)
            starting period of the stimulus onset
        onset_stop
            right before stimulus occurs 
            
        returns: 
            array with the data during the onset period for all trials"""
    
    all_data_for_channel = data_for_trial_list[channel].T
    onset_for_channel = []
    onset_start = int(onset_start)
    onset_stop = int(onset_stop)
    
    for trial in range(len(all_data_for_channel)): 
        one_trial = all_data_for_channel[trial]
        onset_for_channel.append(one_trial[onset_start:onset_stop])
        
    return np.array(onset_for_channel)


def zscore_from_baseline(data_for_trial_list, channel, onset_start, onset_stop):
    
    onset_data = stimulus_onset_one_channel(data_for_trial_list, channel, onset_start, onset_stop)
    
    average_onset = np.average(onset_data)
    standard_dev = np.std(onset_data)
    
    all_data_for_channel = data_for_trial_list[channel].T #returns data for one channel for all trials
    zscored = (all_data_for_channel - average_onset)/(standard_dev)
        
    return zscored    


def get_average_zscore(data_for_trial_list, channel, onset_start, onset_stop):
 
    data_for_channel = zscore_from_baseline(data_for_trial_list, channel, onset_start, onset_stop)
    
    different_format = np.transpose(data_for_channel) #array shape(12000, 60)

    
#     trial_mat = np.zeros((tmax, len(data_for_channel))) #array shape(12000, 60)

    
#     for timepoint in range(len(different_format)): # for each timepoint in all of 60 trials
#         sub_trial = different_format[timepoint] #this gives us the data for the first timepoint across 60 trials when timepoint = 0

#         trial_mat[timepoint, :] = sub_trial
    
    mean_zscore = np.mean(different_format, axis = 1)
    
    return mean_zscore


def get_std(data_for_trial_list, channel, onset_start, onset_stop):
    
    data_for_channel = zscore_from_baseline(data_for_trial_list, channel, onset_start, onset_stop)
    
    different_format = np.transpose(data_for_channel)
    
    std_trial = np.std(different_format, axis=1)
    
    return std_trial


