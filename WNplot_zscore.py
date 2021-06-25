#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 14:40:50 2021

@author: vanessagutierrez
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import tdt
import os
import warnings
warnings.simplefilter("ignore")

#%%
def all_data_one_channel(data_for_trial_list, channel, tmax = 6000):
    '''
    
    Parameters
    ----------
    data_for_trial_list : LIST
        list with data with an array for each trial that contains subarrays for each channel
    channel : INT
        channel you want data for
    tmax : INT, optional
        the default is 6000.

    Returns
    -------
    ARRAY
        array with dimensions (60, tmax) that contains the data for one channel

    '''
    data_for_channel = []
    channel = channel - 1 #subtract 1 because the array is 0 indexed but the channels are not 
    
    for trial in range(len(data_for_trial_list)):
        one_trial = data_for_trial_list[trial][channel]
        data_for_channel.append(one_trial[:tmax])
        
    return np.array(data_for_channel)
#%%
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
    
    onset_for_channel = []
    channel = channel - 1 #subtract 1 because the array is 0 indexed but the channels are not 
    onset_start = int(onset_start)
    onset_stop = int(onset_stop)
    
    for trial in range(len(data_for_trial_list)):
        one_trial = data_for_trial_list[trial][channel]
        onset_for_channel.append(one_trial[onset_start:onset_stop])
        
    return np.array(onset_for_channel)
#%%
def zscore_from_baseline(data_for_trial_list, channel, onset_start, onset_stop):
    
    onset_data = stimulus_onset_one_channel(data_for_trial_list, channel, onset_start, onset_stop)
    
    average_onset = np.average(onset_data)
    standard_dev = np.std(onset_data)
    
    all_data_for_channel = all_data_one_channel(data_for_trial_list, channel) #returns data for one channel for all trials
    zscored = (all_data_for_channel - average_onset)/(standard_dev)
        
    return zscored
#%%
def get_average_zscore(data_for_trial_list, channel, onset_start, onset_stop, tmax = 6000):
 
    data_for_channel = zscore_from_baseline(data_for_trial_list, channel, onset_start, onset_stop)
    
    different_format = np.transpose(data_for_channel) #array shape(6000, 60)

    
#     trial_mat = np.zeros((tmax, len(data_for_channel))) #array shape(6000, 60)

    
#     for timepoint in range(len(different_format)): # for each timepoint in all of 60 trials
#         sub_trial = different_format[timepoint] #this gives us the data for the first timepoint across 60 trials when timepoint = 0

#         trial_mat[timepoint, :] = sub_trial
    
    mean_trial = np.mean(different_format, axis = 1)
    
    return mean_trial #this should be array of size 6000
#%%
def plot_zscores(data_for_trial_list, chs, onset_start, onset_stop, 
                 tmax = 6000, height = 8, width = 16): 
    for channel in np.arange(len(chs)):
        if channel >= height*width:
            return
        plt.subplot(height, width, channel + 1)
        
        correct_channel = chs[channel]
        data_for_channel_zscored = zscore_from_baseline(data_for_trial_list, correct_channel, onset_start, onset_stop)
        average_for_channel = get_average_zscore(data_for_trial_list, correct_channel, onset_start, onset_stop)
        
        plt.plot(average_for_channel, color = 'k', linewidth= 2, zorder = 9)
        plt.title('Channel ' + str(correct_channel), fontsize= 5)
        plt.vlines(.25*fs, ymin = min(data_for_channel_zscored.flatten()), ymax = max(data_for_channel_zscored.flatten()), color = 'darksalmon', zorder = 10)
        for i in np.arange(len(data_for_channel_zscored)):
            plt.plot(data_for_channel_zscored[i], color = (.85,.85,.85), linewidth = 0.5)
        
        plt.xlim(2000, 6000)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
    plt.suptitle('{} Z-Scores Across Channels'.format(animal_block), fontsize=60, y=1)
#%%
def get_data(data_directory, stream, epoc_event):
    '''
    Read data block, extracts stream data, extracts sample rate, extracts data trials.

    Parameters
    ----------
    data_directory : (path)
        Data path
    stream : (str)
        Name of stream in data block
    epoc_event : (str)
        Name of epoc in data block

    Returns
    -------
    stream_data : (np.array)
        array of channels by samples
    fs : (np.float)
        sample rate
    trial_list : (list)
        list of np.arrays (trials), within an array (trial) is channels by samples
    animal_block : (str)
        name of animal block
    '''
    
    tdt_data = tdt.read_block(data_directory)
    animal_block = tdt_data.info.blockname
    stream_data = tdt_data.streams[stream].data
    fs = tdt_data.streams[stream].fs
    tdt_trials = tdt.epoc_filter(tdt_data, epoc_event)
    trial_list = tdt_trials.streams[stream].filtered
    
    return stream_data, fs, trial_list, animal_block
#%%
