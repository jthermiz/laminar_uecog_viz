#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:44:01 2021

@author: macproizzy

Description: 
    
    Allows for plotting of high gamma response across trials for one channel. 
    
    get_trials_mat: put in the tdt data and markers, returns a matrix containing the data from the signal within the 
    timeframe allowed by prebuf and postbuf
    
    zscore_data: zscores the data given in a trials matrix 
    
    compute_high_gamma: computes the high gamma bands for the signal put in as a trials matrix, returns zscored response
    
    get_data: returns the necessary lists and such from the tdt data 
    
    plot_high_gamma: plots the high gamma response during a specific trial for a specific channel 
    
TO DO: create a visualization for visualizing how the high gamma response changes across trials for a channel 
    idea #1: some kind of heat map 
        x-axis: time 
        y-axis: trials
        color: amplitude of the response 
        
    idea #2: plot median amplitude of response across all timepoints 
"""
import matplotlib.pyplot as plt
import numpy as np 
import tdt
import os
import warnings 
from process_nwb.wavelet_transform import wavelet_transform
warnings.simplefilter("ignore")


def get_trials_mat(signal, markers, pre_buf=10000, post_buf=10000):
    """Returns trial matrix
    Args:
        signal (np.array): Signal vector
        markers (list): List of trial onset in samples
        pre_buf (int, optional): Number of samples to pull prior to baseline. Defaults to 10000.
        post_buf (int, optional): Number of samples to pull after. Defaults to 10000.
    Returns:
        trials_mat (np.array): Trial matrix samples by trials
    """
    nsamples = post_buf + pre_buf
    ntrials = len(markers)
    trials_mat = np.empty((nsamples, ntrials))
        
    for idx, marker in enumerate(markers):
        start_frame, end_frame = marker - pre_buf, marker + post_buf
        trials_mat[:, idx] = signal[int(start_frame):int(end_frame)]
    return trials_mat

def zscore_data(tf_data, num_base_pts=200):
    """Compute zscore across trial matrix
    Args:
        tf_data (nparray): Trial matrix of samples x trials
        num_base_pts (int, optional): The first num_base_pts are used for baseline. Defaults to 200.
    Returns:
        tf_norm_data (nparray): Normalized trial matrix
    """
    # Zscore the data
    mean = tf_data[:num_base_pts].mean(axis=0, keepdims=True)
    std = tf_data[:num_base_pts].std(axis=0, keepdims=True)
    tf_norm_data = (tf_data - mean) / std    
    return tf_norm_data


def compute_high_gamma(trials_mat, fs, pre_buf=10000, post_buf=10000, baseline=200):
    """Computes high gamma using wavelet transform
    Args:
        trials_mat (list): List of np-arrays that contain trial matrices (samples x trials)
        fs (numeric): Sample rate
        pre_buf (int, optional): The number of samples to include prior to sample midpoint of trials_mat. Defaults to 10000.
        post_buf (int, optional): The number of samples to include after the sample midpoint. Defaults to 10000.
        baseline (int, optiona): The first set of samples to include in baseline. Defaults to 200.
    Returns:
        Xnorm, f (nparray, nparray): Returns zscored wavelet magnitude coefficients and corresponding frequencies
    """
    Xh, _, f, _  = wavelet_transform(trials_mat, rate=fs, filters='rat', hg_only=True)
    n = Xh.shape[0] // 2
    Xh = Xh[(n - pre_buf):(n + post_buf), :, :] #throw away edges bc of possible effect affects
    #f = f[10:]
    Xm = abs(Xh) #take abs value
    Xnorm = zscore_data(Xm, baseline) #zscore 
    high_gamma = Xnorm.mean(axis = -1)
    return high_gamma


def get_data(data_directory, stream, stim_delay = .25):
    """Gets the data from the tdt file as specified by the stream, the markers for that data, and the frequency
    
    Args: 
        data_directory (string): directory where the data lives 
        stream (string): if ecog, "Wave", if Poly, "Poly"
        stim_delay (float): stimulus onset delay, used to generate a list of marker onsets 
    """
    tdt_data = tdt.read_block(data_directory)
    fs = tdt_data['streams'][stream]['fs']
    fs_stim_delay = 0.25 * fs
    wave_data = tdt_data.streams.Wave.data
    new_wave_data = wave_data.T
    markers = tdt_data.epocs.mark.onset
    marker_onsets = [int(x*fs+fs_stim_delay) for x in markers] 
    return new_wave_data, marker_onsets, fs

def plot_high_gamma(data_directory, channel, trial, stream = 'Wave'):
    new_wave_data, marker_onsets, fs = get_data(data_directory, stream)
    trials_mat_test = get_trials_mat(new_wave_data[:, channel], marker_onsets)
    high_gamma_test = compute_high_gamma(trials_mat_test, fs)
    x_axis = np.linspace(-10000, 10000, 20000)
    
    fig, ax = plt.subplots()
    ax.plot(x_axis, high_gamma_test[:, trial], color = "k")
    ax.set_title("Channel {} High Gamma Response: Trial {}".format(channel, trial))
    ax.set_xlabel("Timepoints (Samples?)")
    ax.set_ylabel("Zscored High Gamma Response Coefficient")
    ax.axvline(fs*.25, ymin = min(high_gamma_test[:, trial]), ymax = max(high_gamma_test[:, trial]), color = 'darksalmon')
    fig

def plot_heatmap(data_directory, channel, stream = 'Wave'):
    new_wave_data, marker_onsets, fs = get_data(data_directory, stream)
    trials_mat_test = get_trials_mat(new_wave_data[:, channel], marker_onsets)
    high_gamma_test = compute_high_gamma(trials_mat_test, fs)
    #x_axis = np.linspace(-10000, 10000, 10000)

    fig, ax = plt.subplots(1,1) 
    ax.set_title("Channel {} High Gamma Response".format(channel))
    ax.set_xlabel("Samples")
    ax.set_ylabel("Trials")
    ax.axvline(10000)
    sns.heatmap(high_gamma_test.T, ax = ax, robust = True)        

import seaborn as sns
data_directory = r'/Users/macproizzy/Desktop/Raw_Signal/RVG02_B01'
plot_high_gamma(data_directory, 13, 25)
plot_heatmap(data_directory, 13)

plot_heatmap(data_directory, 0)
for i in np.arange(1,6):
    plot_heatmap(data_directory, i)




