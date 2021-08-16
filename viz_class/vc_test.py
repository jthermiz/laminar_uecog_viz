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

plot_trials = test.plot_trials(chs_ordered)

test.channel_order
test.animal_block
trials = test.get_all_trials_matrices()
trials[1][:]

for trial in trials.values():
    print(trial)

for i in np.arange(len(test.channel_order)): 
    plt.subplot(8, 16, i + 1)
    plt.rcParams['figure.dpi'] = 100.0
    for trial in trials.values():
        sub_trial = trial[test.channel_order[i],:]
        plt.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
        ymin, ymax = np.min(sub_trial), np.max(sub_trial)
        # plt.plot([stim_delay*self.fs, stim_delay*self.fs], [ymin, ymax], 'darksalmon')
        # plt.xlim(first, last)
        # #     plt.plot([.3*fs, .3*fs], [ymin, ymax], 'darksalmon')
        plt.title('Ch. ' + str(test.channel_order[i]), fontsize= 4.5)
        # plt.xticks([first, first + 1000, last - 1000, last],[-100, 0, 100, 200])
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        # plt.xlabel('time (ms)', fontsize= 4)
        # plt.ylabel('mV', fontsize= 4)
        
    plt.suptitle('{} Average Trial Across Channels'.format(test.animal_block), fontsize=13, y=1)
    

f = plt.figure()
f.set_size_inches(100, 45)
f.supxlabel('time (ms)', fontsize= 4)
f.supylabel('mV', fontsize= 4)
plt.tight_layout()





fig, axs = plt.subplots(8, 16, figsize=(10, 5), constrained_layout=True, sharex=True, sharey=True)


for nn, ax in enumerate(axs.flat):
    ax.set_xlim(9000, 11800)
    for i in np.arange(len(test.channel_order)):
        for trial in trials.values():
            sub_trial = trial[test.channel_order[i],:]
            ax.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
            ymin, ymax = np.min(sub_trial), np.max(sub_trial)
        #     ax.set_title(column, fontsize='small', loc='left')
        #     ax.set_ylim([0, 100])
        #     ax.grid()
        # fig.supxlabel('Year')
        # fig.supylabel('Percent Degrees Awarded To Women')
plt.show()












fig, axs = plt.subplots(8, 16, figsize=(20, 10))
fig.tight_layout()
chs = test.channel_order
trials_dict = trials
idx = 0 #starting point for index in grid 
while idx < (8*16):
    row, col = idx // 16, idx % 16
    #ch = chs[idx]
    ax = axs[row, col]
    channel = chs[idx]
    trials_mat = trials_dict[channel]
    for tidx, trial in enumerate(trials_mat):
        sub_trial = trial[:]
        ymin, ymax = np.min(sub_trial), np.max(sub_trial)
        ax.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
    ax.set_title("Channel {}".format(chs[idx]))
    ax.set_xlabel("Time (ms)")
#       ax.set_ylabel("Frequency (Hz)")
    idx += 1
# fig.suptitle('Neural Spectrogram: {}'.format(animal_block), fontsize = 20, y = 1)
fig















one_channel = test.get_trials_matrix(111)

trials = test.get_all_trials_matrices()
channel = trials[111]

all_trials = test.get_all_trials_matrices()
channel = all_trials[111].T

fig, ax = plt.subplots()
fig.tight_layout()
ax.set_xlim(9000, 11800)
ax.set_xticks([9000, 10000, 11000])
ax.set_xticklabels([-100, 0, 100])

for tidx, trial in enumerate(channel):
    sub_trial = trial[:]
    ymin, ymax = np.min(sub_trial), np.max(sub_trial)
    ax.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
    ax.axvline(x= 10025, ymin=0.05, ymax=0.95, color = 'darksalmon')


trial_mat = np.zeros((20000, len(channel)))

for tidx, trial in enumerate(channel):
    sub_trial = trial[:]
    trial_mat[:, tidx] = sub_trial 
    
mean_trial = np.mean(trial_mat, axis=1)
ax.plot(mean_trial, color='k', linewidth=2.5, zorder=10)

ax.set_title("Channel {} Average Waveform Across Trials".format(channel))
ax.set_xlabel("Time (ms)")
ax.set_ylabel("μV")




test.fs





for i in np.arange(len(chs_ordered)): 
    plt.subplot(height, width, i + 1)

    for tidx, trial in enumerate(trial_list):
        sub_trial = trial[chs[i] - 1,:tmax]
        plt.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
        ymin, ymax = np.min(sub_trial), np.max(sub_trial)
        plt.plot([stim_delay*self.fs, stim_delay*self.fs], [ymin, ymax], 'darksalmon')
        plt.xlim(first, last)
        #     plt.plot([.3*fs, .3*fs], [ymin, ymax], 'darksalmon')
        plt.title('Ch. ' + str(chs[i]), fontsize= 9)
        plt.xticks([first, first + 1000, last - 1000, last],[-100, 0, 100, 200])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('time (ms)', fontsize= 8)
        plt.ylabel('mV', fontsize= 8)
                
    plt.suptitle('{} Average Trial Across Channels'.format(animal_block), fontsize=20, y=1)


















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




