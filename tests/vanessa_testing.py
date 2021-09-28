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

data_directory = r'/Users/vanessagutierrez/data/Rat/RVG14/RVG14_B03'
stream = 'Wave'
stimulus = 'wn2'
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















# channel = [3]

# for i in channel:
#     print(i)

# def plot_zscore(trials_dict, fs, channel, stim_duration, onset_start, onset_stop, fig = None, ax = None, labels = True):
    
#     trials_mat = trials_dict[channel]
        
#     if fig == None and ax == None:
#         fig, ax = plt.subplots()
#     else:
#         fig, ax = fig, ax 
        
#     fig.tight_layout()
#     # rect = (0.04, 0.04, 1, 0.95)    
#     x_axis = np.linspace(-10000, 10000, len(trials_mat))
#     # x_axis = np.linspace(-5000, 500, len(trials_mat)) #wave tonediag
#     #x_axis = np.linspace(-10000, 5500, 20000)  #poly tonediag
#     x_axis = (x_axis/fs) * 1000
    
#     ax.set_xlim(-150, 150)
#     # ax.set_xlim(-100, 100)
    
#     #data_for_channel_zscored = gz.zscore_data(trials_mat)
#     #average_for_channel = np.mean(data_for_channel_zscored, axis = 1)
        
#     stim_stop = stim_duration *1000
            
#     data_for_channel_zscored = gz.zscore_from_baseline(trials_dict, channel, onset_start, onset_stop)
#     average_for_channel = gz.get_average_zscore(trials_dict, channel, onset_start, onset_stop)
    
#     ax.set_ylim(min(average_for_channel)-.5, max(average_for_channel)+.5)    
    
    
#     zscored_data = data_for_channel_zscored.T
    
    
#     num_trials = zscored_data.shape[1]
#     mean = np.mean(zscored_data, axis = -1)
            
#     standard_dev = np.std(zscored_data, axis = -1)
#     sqrt_n = np.sqrt(num_trials)
#     standard_error = standard_dev/sqrt_n
#     ax.fill_between(x_axis, mean - standard_error, mean + standard_error, color = 'k', alpha = .3)
                
#     ax.plot(x_axis, average_for_channel, color = 'k', linewidth= 2, zorder = 9)
#     ax.axvline(x= 0, ymin=min(data_for_channel_zscored.flatten()), ymax=max(data_for_channel_zscored.flatten()), color = 'darksalmon', zorder = 11, linestyle='--')
            
#     ax.axvline(x= stim_stop, ymin=min(zscored_data.flatten()), ymax=max(zscored_data.flatten()), color = 'darksalmon', zorder = 11, linestyle='--')
    
#     ax.set_title("Channel {} Average Zscored Trial".format(channel), fontsize = 15)
    
#     if labels == True:
#         ax.set_xlabel("Time (ms)")
#         ax.set_ylabel("mV")