#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:15:17 2021

@author: vanessagutierrez
"""

from laminar_uecog_viz import data_reader as dr
from laminar_uecog_viz import plot_zscore as pltz
from laminar_uecog_viz import utils
from laminar_uecog_viz import get_zscore as gz
from laminar_uecog_viz import plot_trials

import numpy as np
import matplotlib.pyplot as plt

data_directory = r'/Users/vanessagutierrez/data/Rat/RVG21/RVG21_B02'
stream = 'Wave'
stimulus = 'tone_diagnostic'
channel_order = [
        81, 83, 85, 87, 89, 91, 93, 95, 97, 105, 98, 106, 114, 122, 113, 121,
        82, 84, 86, 88, 90, 92, 94, 96, 99, 107, 100, 108, 116, 124, 115, 123,
        66, 68, 70, 72, 74, 76, 78, 80, 101, 109, 102, 110, 118, 126, 117, 125,
        65, 67, 69, 71, 73, 75, 77, 79, 103, 111, 104, 112, 120, 128, 119, 127,
        63, 61, 59, 57, 55, 53, 51, 49, 25, 17, 26, 18, 10, 2, 9, 1,
        64, 62, 60, 58, 56, 54, 52, 50, 27, 19, 28, 20, 12, 4, 11, 3,
        48, 46, 44, 42, 40, 38, 36, 34, 29, 21, 30, 22, 14, 6, 13, 5,
        47, 45, 43, 41, 39, 37, 35, 33, 31, 23, 32, 24, 16, 8, 15, 7
        ]
poly_channel_order = [ 
                    21, 3, 
                    23, 5, 
                    24, 7, 
                    30, 9, 
                    29, 11, 
                    16, 13, 
                    18, 15, 
                    20, 31, 
                    27, 14, 
                    19, 12, 
                    17, 10, 
                    25, 8, 
                    26, 6, 
                    32, 4, 
                    28, 2, 
                    22, 64, 
                    1, 62, 
                    58, 60, 
                    54, 56, 
                    50, 52,
                    51, 34, 
                    55, 53, 
                    59, 57, 
                    63, 61, 
                    42, 44, 
                    35, 41, 
                    49, 36, 
                    45, 47, 
                    46, 38, 
                    40, 48, 
                    33, 39, 
                    43, 37,
                    ]

rd = dr.data_reader(data_directory, stream, stimulus)
signal_data, fs, stim_markers, animal_block = rd.get_data()
marker_onsets, stim_duration = rd.get_stim_onsets()

trials_dict = utils.get_all_trials_matrices(signal_data, marker_onsets, channel_order)

onset_start = int((stim_duration-0.05)*fs)
onset_stop = int((stim_duration+0.05)*fs)

pltz.plot_zscore1(trials_dict, fs, 3, stim_duration, onset_start, onset_stop, fig = None, ax = None)

pltz.plot_zscore2(trials_dict, fs, 3, stim_duration, fig = None, ax = None)



plot_trials.plot_trials_matrix(trials_dict, fs, channel_order, stream)

plot_trials.plot_trials(trials_dict, fs, 3, stream)





import numpy as np 
from laminar_uecog_viz import data_reader as dr
from laminar_uecog_viz import utils
from pynwb import NWBHDF5IO

data_directory = r'/Users/vanessagutierrez/data/Rat/RVG21/RVG21_B02'
stream = 'Wave'
stimulus = 'tone_diagnostic'

rd = dr.data_reader(data_directory, stream, stimulus)
signal_data, fs, stim_markers, animal_block = rd.get_data()
marker_onsets, stim_duration = rd.get_stim_onsets()

channel_order = [
        81, 83, 85, 87, 89, 91, 93, 95, 97, 105, 98, 106, 114, 122, 113, 121,
        82, 84, 86, 88, 90, 92, 94, 96, 99, 107, 100, 108, 116, 124, 115, 123,
        66, 68, 70, 72, 74, 76, 78, 80, 101, 109, 102, 110, 118, 126, 117, 125,
        65, 67, 69, 71, 73, 75, 77, 79, 103, 111, 104, 112, 120, 128, 119, 127,
        63, 61, 59, 57, 55, 53, 51, 49, 25, 17, 26, 18, 10, 2, 9, 1,
        64, 62, 60, 58, 56, 54, 52, 50, 27, 19, 28, 20, 12, 4, 11, 3,
        48, 46, 44, 42, 40, 38, 36, 34, 29, 21, 30, 22, 14, 6, 13, 5,
        47, 45, 43, 41, 39, 37, 35, 33, 31, 23, 32, 24, 16, 8, 15, 7
        ]


io = NWBHDF5IO('/Users/vanessagutierrez/Desktop/NWB_Test/RVG21/RVG21_B02.nwb', 'r')
nwb = io.read()

nwb_signal_data = nwb.acquisition['ECoG'].data[:]

fs = nwb.acquisition['ECoG'].rate

trials_df = nwb.trials.to_dataframe()

new_signal_dataa = utils.channel_orderer(nwb_signal_data, channel_order)


t = np.arange(0, tend-tbeg)/fs_final

def nwb_stim_t(trials_df, fs):
    
    df_s = trials_df[trials_df["sb"] == "s"]
    
    onsets = df_s.iloc[:, [0,2]]
    stim_markers = onsets['start_time'].to_list()
    stim_onsets = [int(x*fs) for x in stim_markers]
    stim_start_times = np.array(stim_onsets)
    
    offsets = df_s.iloc[:, [1,2]]
    stim_mrks = offsets['stop_time'].to_list()
    stim_offsets = [int(x*fs) for x in stim_mrks]
    stim_stop_times = np.array(stim_offsets)
    
    return stim_start_times, stim_stop_times


def nwb_baseline_t(trials_df, fs):

    df_b = trials_df[trials_df["sb"] == "b"]
    
    b_onsets = df_b.iloc[:, [0,2]]
    base_markers = b_onsets['start_time'].to_list()
    base_onsets = [int(x*fs) for x in base_markers]
    base_start_times = np.array(base_onsets)
    
    b_offsets = df_b.iloc[:, [1,2]]
    base_mrks = b_offsets['stop_time'].to_list()
    base_offsets = [int(x*fs) for x in base_mrks]
    base_stop_times = np.array(base_offsets)
    
    return base_start_times, base_stop_times
    
stim_start_times, stim_stop_times = nwb_stim_t(trials_df, fs)    
base_start_times, base_stop_times = nwb_baseline_t(trials_df, fs)

def get_trials(X, t, stim_start_times, stop_times, baseline_start_times, baseline_stop_times):
    """Convert N-D array into N-D + 1 set of trial arrays

    Parameters
    ----------
    X : N-D array
        [description]
    t : 1-D array
        Array of time points in seconds
    start_times : array_like
        Trial start times in seconds
    stop_times : array_like
        Trial stop times in seconds
    baseline_start_times : array_like
        Start times for calculating baseline normalizing statistics in seconds
    baseline_stop_times : array_like
        Stop times for calculating baseline normalizing statistics in seconds

    Raises
    ------
    NotImplementedError
        [description]
    """
    raise NotImplementedError
    
    nsamples = post_buf + pre_buf
    ntrials = len(stim_start_times)
    trials_mat = np.empty((nsamples, ntrials))
    channel_data = signal_data[:, channel]
    
    for idx, onset in enumerate(stim_start_times):
        start_frame, end_frame = onset - pre_buf, onset + post_buf
        trials_mat[:, idx] = channel_data[int(start_frame):int(end_frame)]
    return trials_mat
    


def get_ch_trials_matrix(signal_data, marker_onsets, channel, pre_buf = 10000, post_buf = 10000):
    
    """
    Creates a trials matrix for one channel
    
    Parameters
    ----------
    signal_data (np.array): signal data (nsamples, nchannels).
    marker_onsets (list): List of trial sitmulus onsets in samples.
    channnel (int): Specific channel you want data for 
    pre_buf (int, optional): Number of samples to pull prior to baseline. Defaults to 10000.
    post_buf (int, optional): Number of samples to pull after. Defaults to 10000.
    
    Returns
    -------
    trials_mat (np.array): Trial matrix for one channel (samples, trials).
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
    Python dictionary where the key is the channel and the value is the trial matrix for that channel
    now, instead of calling get_trials_matrix a multiple of times when we want to visualize, we can 
    iterate over the keys in the all_trials matrix 

    Parameters
    ----------
    signal_data (np.array): signal data (nsamples, nchannels).
    marker_onsets (list): List of trial sitmulus onsets in samples.
    channel_order (list): List of channel order on ECoG array or laminar probe.
    pre_buf (int, optional): Number of samples to pull prior to baseline. Defaults to 10000.
    post_buf (int, optional): Number of samples to pull after. Defaults to 10000.

    Returns
    -------
    trials_dict (dict): Trials dictionary that is the length of channel_order. Each channel key has its trial matrix (samples, trials).
    """
    
    all_trials = {}
    for i in np.arange(len(channel_order)):
        one_channel = get_ch_trials_matrix(signal_data, marker_onsets, i)
        all_trials[channel_order[i]] = one_channel
    trials_dict = all_trials
    return trials_dict







