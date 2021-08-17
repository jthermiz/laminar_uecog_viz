#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:34:23 2021

@author: macproizzy
"""

import viz_class as vc
import numpy as np
import matplotlib.pyplot as plt

RVG06_B03 = r'/Users/vanessagutierrez/data/Rat/RVG06/RVG06_B03'
stream = 'Wave'
stimulus = 'wn2'
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

test = vc.viz(RVG06_B03, stream, stimulus)

channels = test.channel_order
test.animal_block

test.plot_trials(channels)

plot_trials = test.plot_trials(channels)


plot_trials = test.plot_trials(109)

trials = test.get_all_trials_matrices()






















# fig, axs = plt.subplots(8, 16, figsize=(60, 30), sharex=True)
# fig.tight_layout(rect=[0.035, 0.03, 0.95, 0.97])
# fig.supxlabel('Time (ms)', fontsize = 45)
# fig.supylabel('μV', fontsize = 45)
# fig.suptitle('{} Average Trial Across Channels'.format(test.animal_block), fontsize = 55, y = 1)
# chs = test.channel_order
# # trials_dict = trials
# idx = 0 #starting point for index in grid 
# while idx < (8*16):
#     row, col = idx // 16, idx % 16
#     #ch = chs[idx]
#     ax = axs[row, col]
#     channel = chs[idx]
#     trials_mat = trials[channel].T
#     trial_mat = np.zeros((20000, len(trials_mat)))
#     ax.set_xlim(9000, 11800)
#     ax.set_xticks([9000, 10000, 11000])
#     ax.set_xticklabels([-100, 0, 100])
#     for tidx, trial in enumerate(trials_mat):
#         sub_trial = trial[:]
#         trial_mat[:, tidx] = sub_trial 
#         ymin, ymax = np.min(sub_trial), np.max(sub_trial)
#         ax.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
#     mean_trial = np.mean(trial_mat, axis=1)
#     ax.plot(mean_trial, color='k', linewidth=2.5, zorder=10)

#     ax.set_title("Channel {}".format(chs[idx]))
#     # ax.set_xlabel("Time (ms)")
# #       ax.set_ylabel("Frequency (Hz)")
#     idx += 1
# # fig.supxlabel('Time (ms)', fontsize = 30)
# # fig.supylabel('μV', fontsize = 30)
# fig




# one_channel = test.get_trials_matrix(111)

# trials = test.get_all_trials_matrices()
# channel = trials[chs_ordered[5]]

# all_trials = test.get_all_trials_matrices()
# channel = all_trials[111].T

# fig, ax = plt.subplots()
# fig.tight_layout()
# ax.set_xlim(9000, 11800)
# ax.set_xticks([9000, 10000, 11000])
# ax.set_xticklabels([-100, 0, 100])


# for tidx, trial in enumerate(channel):
#     sub_trial = trial[:]
#     ymin, ymax = np.min(sub_trial), np.max(sub_trial)
#     ax.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
#     ax.axvline(x= 10025, ymin=0.05, ymax=0.95, color = 'darksalmon')


# trial_mat = np.zeros((20000, len(channel)))

# for tidx, trial in enumerate(channel):
#     sub_trial = trial[:]
#     trial_mat[:, tidx] = sub_trial 
    
# mean_trial = np.mean(trial_mat, axis=1)
# ax.plot(mean_trial, color='k', linewidth=2.5, zorder=10)

# ax.set_title("Channel {} Average Waveform Across Trials".format(channel))
# ax.set_xlabel("Time (ms)")
# ax.set_ylabel("μV")







test.plot_trials()
test.plot_spectrogram_matrix(8,16)
test.plot_high_gamma_matrix(8,16)

RVG08_B01 = r'/Users/macproizzy/Desktop/RVG_Data/RVG08_B01'
RVG08 = vc.viz(RVG08_B01, chs_ordered, stream)
RVG08.get_all_trials_matrices()

RVG08.plot_trials()
RVG08.plot_spectrogram_matrix(8,16)
RVG08.plot_high_gamma_matrix(8,16)

RVG13_B4 = r'/Users/macproizzy/Desktop/RVG_Data/RVG13_B4'
RVG134 = vc.viz(RVG13_B4, chs_ordered, stream)
RVG134.get_all_trials_matrices()

RVG134.plot_trials()
RVG134.plot_spectrogram_matrix(8,16)
RVG134.plot_high_gamma_matrix(8,16)


RVG14_B1 = r'/Users/macproizzy/Desktop/RVG_Data/RVG14_B1'
RVG14 = vc.viz(RVG14_B1, chs_ordered, stream)
RVG14.get_all_trials_matrices()

RVG14.plot_trials()
RVG14.plot_spectrogram_matrix(8,16)
RVG14.plot_high_gamma_matrix(8,16)
#type(test.trials_dict)

#est.plot_all_hg_response(83)

#test.plot_high_gamma_matrix(2, 3)
#test.plot_spectrogram_matrix(2,3)




