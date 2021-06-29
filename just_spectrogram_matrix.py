#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:14:41 2021

@author: macproizzy

Change to log based scale 

Description: 
    Based on just_high_gamma plotting script, takes in data, creates trials matrix, unscrambles matrix, plots 
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import tdt
import os
from process_nwb.wavelet_transform import wavelet_transform
import tdt as tdt

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


def get_trials_mat(signal, markers, pre_buf=10000, post_buf=10000):
    """Returns trial matrix
    Args:
        signal (np.array): Signal vector (for one channel)
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


def channel_orderer(signal_data, correct_channel_order):
    """Puts the wave data into the order of the channels
    Args: 
        data: signal data in timepoints x channels
        chs (list): the correct order of the channels"""
    shape_wanted = signal_data.shape
    new_data = np.empty((shape_wanted[0], shape_wanted[1]))
    
    for i in np.arange(shape_wanted[1]):
        new_data[:, i] = signal_data[:, (correct_channel_order[i] - 1)]
        print("Data for channel {} is now at index {}".format(correct_channel_order[i] - 1, i))
    return new_data


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
    Xh = Xh[:, :, 10:] #throw away low frequencies
    f = f[10:]
    Xm = abs(Xh) #take abs value
    Xnorm = zscore_data(Xm, baseline) #zscore 
    return Xnorm, f


def plot_spectrogram(tf_data, f, tmin, tmax, colorbar=False, ax=None, fig=None, zero_flag=False, log_scale=True):
    """Plots spectrogram
    Args:
        tf_data (nparray): Trial matrix samples x trials
        f (nparray): Frequency vector
        tmin (np.float): X-axis min display time (whatever is specified is what the x-axis will be label irrespective if it is correct)
        tmax (np.float): X-axis max display time (whatever is specified is what the x-axis will be label irrespective if it is correct)
        colorbar (bool, optional): Whether to plot colorbar. Defaults to False.
        ax (class, optional): Matplotlib axes handle. Defaults to None.
        fig (class, optional): Matplotlib figure handle. Defaults to None.
        zero_flag (boolean, optional): Plots red line at t=0
        log_scale (boolean, optional): Plots y-scale in base 2
    """
    #tf_data: samples x frequencies
    if (ax is None) or (fig is None):
        fig, ax = plt.subplots(1, 1)
        
    if log_scale:
        aspect = 10
    else:
        aspect = 1/5
        
    pos = ax.imshow(tf_data.T, interpolation='none', aspect=aspect, cmap='binary', 
                    origin='lower', extent=[tmin, tmax, f[0], f[-1]])
    if log_scale:
        ax.set_yscale('symlog', basey=2)
    if zero_flag:
        ax.plot([0 ,0], [f[0], f[-1]], 'r')
    if colorbar:
        fig.colorbar(pos, ax=ax, shrink=0.7, pad = 0.02)
        
        
def plot_spectrogram_matrix(data, fs, markers, chs, nrow, ncol, pre_buf=10000, post_buf=10000):
    """Extracts trials, compute wavelet coeffients, takes median across trials and plot spectrogram matrix
    Args:
        data (nparray): Data matrix samples x channels
        f (np.array): Frequency vector
        fs (numeric): Sample rate
        markers (list): List of trial onset times in samples
        nrow (int): Number of rows
        ncol (int): Number of columns
        pre_buf (int): Number of samples to pull prior to stimulus onset
        post_buf (int): Number of samples to pull after stimulus onset
    """
    fig, axs = plt.subplots(nrow, ncol, figsize=(20, 10))
    fig.tight_layout()
    idx = 0 #starting point for index in grid 
    while idx < (nrow*ncol):
        row, col = idx // ncol, idx % ncol
        #ch = chs[idx]
        ax = axs[row, col]
        trials_mat = get_trials_mat(data[:, idx], markers, pre_buf=pre_buf, post_buf=post_buf)
        tf_data, f = compute_spectrogram(trials_mat, fs)
        tf_data = np.median(tf_data, axis=1)
        plot_spectrogram(tf_data, f, -10, 100, ax=ax, fig=fig, colorbar=True,log_scale=False)
        ax.set_title("Channel {}".format(chs[idx]))
        ax.set_xlabel("Time (ms)")
#        ax.set_ylabel("Frequency (Hz)")
        idx += 1
    fig
    
    
##### vars to run script
data_directory = r'/Users/vanessagutierrez/Desktop/Rat/RVG08/RVG08_B1'
stream = 'Wave'
chs_ordered = [
       81, 83, 85, 87, 89, 91, 93, 95, 97, 105, 98, 106, 114, 122, 113, 121,
       82, 84, 86, 88, 90, 92, 94, 96, 99, 107, 100, 108, 116, 124, 115, 123,
       66, 68, 70, 72, 74, 76, 78, 80, 101, 109, 102, 110, 118, 126, 117, 125,
       65, 67, 69, 71, 73, 75, 77, 79, 103, 111, 104, 112, 120, 128, 119, 127,
       63, 61, 59, 57, 55, 53, 51, 49, 25, 17, 26, 18, 10, 2, 9, 1,
       64, 62, 60, 58, 56, 54, 52, 50, 27, 19, 28, 20, 12, 4, 11, 3,
       48, 46, 44, 42, 40, 38, 36, 34, 29, 21, 30, 22, 14, 6, 13, 5,
       47, 45, 43, 41, 39, 37, 35, 33, 31, 23, 32, 24, 16, 8, 15, 7
       ]

#stream = 'Poly'
#chs_ordered = [ 
#            27, 37,
#            26, 38, 
#            25, 39, 
#            24, 40, 
#            23, 41,
#            22, 42, 
#            21, 43, 
#            20, 44, 
#            19, 45, 
#            18, 46,
#            17, 47, 
#            16, 48, 
#            15, 49, 
#            14, 50, 
#            13, 51,
#            12, 52, 
#            11, 53, 
#            10, 54, 
#            9, 55, 
#            8, 56, 
#            7, 57, 
#            6, 58, 
#            5, 59, 
#            4, 60, 
#            3, 61, 
#            2, 62, 
#            1, 63, 
#            28, 64, 
#            29, 36, 
#            30, 35, 
#            31, 34, 
#            32, 33,
#            ]

new_wave_data, marker_onsets, fs = get_data(data_directory, stream)
#one_channel = new_wave_data[:, 13]
#trials_mat = get_trials_mat(one_channel, marker_onsets)
unscrambled = channel_orderer(new_wave_data, chs_ordered)
plot_spectrogram_matrix(unscrambled, fs, marker_onsets, chs_ordered, nrow = 2, ncol = 2)



    
