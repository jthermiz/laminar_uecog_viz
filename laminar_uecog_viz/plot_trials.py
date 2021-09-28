#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:31:36 2021

@author: vanessagutierrez
"""

import numpy as np
import matplotlib.pyplot as plt
from laminar_uecog_viz import utils

def plot_trials(trials_dict, fs, channel, stream, fig = None, ax = None, labels = True): 
    '''
    Plot data trials and mean of trials per channel.
  
    Parameters
    ----------
    trial_list : (list)
        list of np.arrays (trials), within an array (trial) is channels by samples
    stream : (str)
        Name of stream in data block
    fs : TYPE
        DESCRIPTION.
    trials : (plot, optional)
        Whether to plot all trials & mean of all trials. Defaults to True.
    '''

    trials_mat = trials_dict[channel].T
    
    if fig == None and ax == None:
        fig, ax = plt.subplots()
        fig.tight_layout()
    else:
        # fig, ax = fig, ax 
        ax = ax 
        #fig = fig
        
    x_axis = np.linspace(-10000, 10000, len(trials_mat.T))
    x_axis = (x_axis/fs) * 1000
    
    ax.set_xlim(-150, 150)
    
    trial_mat = np.zeros((20000, len(trials_mat)))
    
    for tidx, trial in enumerate(trials_mat):
        sub_trial = trial[:]
        trial_mat[:, tidx] = sub_trial
        ax.plot(x_axis, sub_trial, color = (.85, .85, .85), linewidth = 0.5)
        
    # ax.axvline(x = 0, ymin=min(trial_mat.flatten()), ymax=max(trial_mat.flatten()), color = 'darksalmon', zorder = 11, linestyle='--')
    utils.plot_markers(ax, [0,.1])
    
    mean_trial = np.mean(trial_mat, axis=1)
    ax.plot(x_axis, mean_trial, color='k', linewidth=2, zorder=10)
    
    if labels == True:
        ax.set_xlabel("Time (ms)")
        ax.set_title("Channel {} Average Across Trials".format(channel), fontsize = 15)
        if stream == "Wave" or "ECoG":
            ax.set_ylabel("μV")
        if stream == "Poly":
            ax.set_ylabel("mV")
            
    # fig.savefig("{}/{}_Average_Trial_Ch.{}_{}.png".format(self.savepic, self.animal_block, channel, self.stream), dpi=300)
    

    
    
def plot_trials_matrix(trials_dict, fs, channel_order, stream):
    
    if stream == 'ECoG' or 'Wave':
        axes = utils.ecog_axes(channel_order, plot_type='Trials')
    elif stream == 'Poly':
        axes = utils.poly_axes(channel_order, plot_type='Trials')
    else:
        raise ValueError("`stream` must be one of 'ECoG' or 'Poly'")
    
    for ch, ax in zip(channel_order, axes):
        plot_trials(trials_dict, fs, ch, stream, fig = None, ax = ax, labels = False)     

            
          
            
          
