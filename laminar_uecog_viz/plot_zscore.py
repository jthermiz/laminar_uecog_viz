#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:56:48 2021

@author: vanessagutierrez
"""
import numpy as np
import matplotlib.pyplot as plt
from laminar_uecog_viz import get_zscore as gz
from laminar_uecog_viz import utils 
# from viz import get_all_trials_matrices as gtm


def plot_zscore1(trials_dict, fs, channel, stim_duration, onset_start, onset_stop, fig = None, ax = None, labels = True):
    """
    
    Parameters
    ----------
    trials_dict (dict): Trials dictionary that is the length of channel_order. Each channel key has its trial matrix (samples, trials).
    fs (np.float): sample rate
    channel (int): Specific channel you want data for 
    stim_duration (int): duration of stimulus in seconds
    onset_start (int): starting period of the stimulus onset including some ms of baseline before.
    onset_stop (int): ending period of the stimulus onset including some ms of baseline after.
    fig (class, optional): Matplotlib figure handle. Defaults to None.
    ax (class, optional): Matplotlib axes handle. Defaults to None.
    labels (optional): Whether you want axes labels.The default is True.

    """
    
    trials_mat = trials_dict[channel]
        
    if fig == None and ax == None:
        fig, ax = plt.subplots()
        fig.tight_layout()
    else:
        # fig, ax = fig, ax 
        ax = ax 
        #fig = fig
        
    
    # rect = (0.04, 0.04, 1, 0.95)    
    x_axis = np.linspace(-10000, 10000, len(trials_mat))
    # x_axis = np.linspace(-5000, 500, len(trials_mat)) #wave tonediag
    #x_axis = np.linspace(-10000, 5500, 20000)  #poly tonediag
    x_axis = (x_axis/fs) * 1000
    
    ax.set_xlim(-150, 150)
    # ax.set_xlim(-100, 100)
    
    #data_for_channel_zscored = gz.zscore_data(trials_mat)
    #average_for_channel = np.mean(data_for_channel_zscored, axis = 1)
        
    stim_stop = stim_duration *1000
            
    data_for_channel_zscored = gz.zscore_from_baseline(trials_dict, channel, onset_start, onset_stop)
    average_for_channel = gz.get_average_zscore(trials_dict, channel, onset_start, onset_stop)
    
    ax.set_ylim(min(average_for_channel)-.5, max(average_for_channel)+.5)    
    
    
    zscored_data = data_for_channel_zscored.T
    
    
    num_trials = zscored_data.shape[1]
    mean = np.mean(zscored_data, axis = -1)
            
    standard_dev = np.std(zscored_data, axis = -1)
    sqrt_n = np.sqrt(num_trials)
    standard_error = standard_dev/sqrt_n
    ax.fill_between(x_axis, mean - standard_error, mean + standard_error, color = 'k', alpha = .3)
                
    ax.plot(x_axis, average_for_channel, color = 'k', linewidth= 2, zorder = 9)
    ax.axvline(x= 0, ymin=min(data_for_channel_zscored.flatten()), ymax=max(data_for_channel_zscored.flatten()), color = 'darksalmon', zorder = 11, linestyle='--')
            
    ax.axvline(x= stim_stop, ymin=min(zscored_data.flatten()), ymax=max(zscored_data.flatten()), color = 'darksalmon', zorder = 11, linestyle='--')
    
   
    
    if labels == True:
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("mV")
        ax.set_title("Channel {} Average Zscored Trial".format(channel), fontsize = 15)
    
            
    # fig.savefig("{}/{}_Zscore_Ch.{}_{}.png".format(self.savepic, self.animal_block, channel, self.stream), dpi=300)
    

def plot_zscore2(trials_dict, fs, channel, stim_duration, fig = None, ax = None, labels = True, num_base_pts=600, std_error = True, trials = False):
    """
    
    Parameters
    ----------
    trials_dict (dict): Trials dictionary that is the length of channel_order. Each channel key has its trial matrix (samples, trials).
    fs (np.float): sample rate
    channel (int): Specific channel you want data for 
    stim_duration (int): duration of stimulus in seconds
    onset_start (int): starting period of the stimulus onset including some ms of baseline before.
    onset_stop (int): ending period of the stimulus onset including some ms of baseline after.
    fig (class, optional): Matplotlib figure handle. Defaults to None.
    ax (class, optional): Matplotlib axes handle. Defaults to None.
    labels (optional): Whether you want axes labels.The default is True.

    """
    
    trials_mat = trials_dict[channel]
        
    if fig == None and ax == None:
        fig, ax = plt.subplots()
        fig.tight_layout()
    else:
        # fig, ax = fig, ax 
        ax = ax 
        #fig = fig
        
    
    # rect = (0.04, 0.04, 1, 0.95)    
    x_axis = np.linspace(-5000, 5000, len(trials_mat))
    # x_axis = np.linspace(-5000, 500, len(trials_mat)) #wave tonediag
    #x_axis = np.linspace(-10000, 5500, 20000)  #poly tonediag
    x_axis = (x_axis/fs) * 1000
    
    ax.set_xlim(-150, 150)
    # ax.set_xlim(-100, 100)
    
    data_for_channel_zscored = gz.zscore_data(trials_mat, num_base_pts)
    average_for_channel = np.mean(data_for_channel_zscored, axis = 1)
        
    stim_stop = stim_duration *1000
   
    
    ax.set_ylim(min(average_for_channel)-.7, max(average_for_channel)+.7)    
    
    
    zscored_data = data_for_channel_zscored.T
    
    if std_error:
        num_trials = data_for_channel_zscored.shape[1]
        mean = np.mean(data_for_channel_zscored, axis = -1)
                
        standard_dev = np.std(data_for_channel_zscored, axis = -1)
        sqrt_n = np.sqrt(num_trials)
        standard_error = standard_dev/sqrt_n
        ax.fill_between(x_axis, mean - standard_error, mean + standard_error, color = 'k', alpha = .3)
    
    if trials:        
        trial_mat = np.zeros((len(trials_mat), trials_mat.shape[1]))
        
        for tidx, trial in enumerate(trials_mat.T):
            sub_trial = trial[:]
            trial_mat[:, tidx] = sub_trial
            ax.plot(x_axis, sub_trial, color = (.85, .85, .85), linewidth = 0.5)
    
    ax.plot(x_axis, average_for_channel, color = 'k', linewidth= 2, zorder = 9)
    ax.axvline(x= 0, ymin=min(zscored_data.flatten()), ymax=max(zscored_data.flatten()), color = 'darksalmon', zorder = 11, linestyle='--')
            
    ax.axvline(x= stim_stop, ymin=min(zscored_data.flatten()), ymax=max(zscored_data.flatten()), color = 'darksalmon', zorder = 11, linestyle='--')
    
   
    
    if labels == True:
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("mV")
        ax.set_title("Channel {} Average Zscored Trial".format(channel), fontsize = 15)
    
            
    # fig.savefig("{}/{}_Zscore_Ch.{}_{}.png".format(self.savepic, self.animal_block, channel, self.stream), dpi=300)

    

def plot_zscore_matrix(trials_dict, fs, stim_duration, channel_order, onset_start, onset_stop, device):
    """

    Parameters
    ----------
    trials_dict (dict): Trials dictionary that is the length of channel_order. Each channel key has its trial matrix (samples, trials).
    fs (np.float): Sample rate
    stim_duration (int): Duration of stimulus in seconds
    channel_order (list): List of channel order on ECoG array or laminar probe.
    onset_start (int): starting period of the stimulus onset including some ms of baseline before.
    onset_stop (int): ending period of the stimulus onset including some ms of baseline after.
    device (str, optional): Electrode device, either ECoG or polytrode.

    Raises
    ------
    ValueError: `device` must be one of 'ECoG' or 'Poly'

    """
    if device == 'ECoG':
        axes = utils.ecog_axes(channel_order, plot_type='Z-Score')
    elif device == 'Poly':
        axes = utils.poly_axes(channel_order, plot_type='Z-Score')
    else:
        raise ValueError("`device` must be one of 'ECoG' or 'Poly'")
    
    for ch, ax in zip(channel_order, axes):
        plot_zscore1(trials_dict, fs, ch, stim_duration, onset_start, onset_stop, fig = None, ax = ax, labels = False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def plot_zscore_matrix(trials_dict, fs, stim_duration, nrow, ncol, channel_order):
    
#     chs = channel_order
#     figsize = (40, 20)
#     titlesize = 50
#     labelsize = 30
       
#     fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
#     fig.tight_layout()
#     idx = 0 #starting point for index in grid
    
#     while idx < (nrow*ncol):
#         row, col = idx // ncol, idx % ncol
#         ax = axs[row, col]
#         plot_zscore(trials_dict, fs, chs[idx], stim_duration, fig = fig, ax = ax, labels = False)
#         idx += 1
    
#     # animal_block = self.animal_block
#     # .format(animal_block)
#     fig.suptitle('Z-Scored Responses Across Channels', fontsize = titlesize, y = 1)
#     fig.supylabel('mV', fontsize = labelsize)
#     fig.supxlabel('Time(ms)', fontsize = labelsize)
    

