#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:44:17 2021

@author: macproizzy

Script to translate raw data into an object that can be easily visualized for the following plots: 
    average and standard error 
    high gamma response 
    neural spectrograms
"""

import matplotlib.pyplot as plt
import numpy as np 
import tdt
import os
import warnings 
import seaborn as sns
from process_nwb.wavelet_transform import wavelet_transform
from data_reader import data_reader
warnings.simplefilter("ignore")


class viz:
    
    def __init__(self, data_directory, stream, stimulus):
        """
        Init: create an instance of the viz class 

        Parameters
        ----------
        data_directory : TYPE
            DESCRIPTION.
        channel_order : TYPE
            DESCRIPTION.
        stream : TYPE
            DESCRIPTION.

        Returns
        -------
        an instance of the viz class, which allows plotting of data
        
        Attributes:
            data_directory: directory where data lives
            channel_order: list with the correct order of channels
            stream: wave or poly
            tdt_data: data before processing
            fs = sampling frequency 
            fs_stim_delay: stimulus delay
            wave_data: all of the data before processing in format of timepoints x channels
                **note that the index of the array does not match channel number
            markers: markers without any manipulations
            marker_onsets: markers with fs stim delay taken into consideration

        """
        #added stimulus parameter for the data reader class. 
        self.data_directory = data_directory
        self.stream = stream
        self.stimulus = stimulus
        if self.stream == "Wave" or "ECoG":
            height = 8 
            width = 16
            tmax = 6000
            first, last = 2000, 5000
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
        if self.stream == "Poly":
            height = 32 
            width = 2
            tmax = 12000
            first, last = 5500, 7000
            channel_order = [ 
                    27, 37,
                    26, 38, 
                    25, 39, 
                    24, 40, 
                    23, 41,
                    22, 42, 
                    21, 43, 
                    20, 44, 
                    19, 45, 
                    18, 46,
                    17, 47, 
                    16, 48, 
                    15, 49, 
                    14, 50, 
                    13, 51,
                    12, 52, 
                    11, 53, 
                    10, 54, 
                    9, 55, 
                    8, 56, 
                    7, 57, 
                    6, 58, 
                    5, 59, 
                    4, 60, 
                    3, 61, 
                    2, 62, 
                    1, 63, 
                    28, 64, 
                    29, 36, 
                    30, 35, 
                    31, 34, 
                    32, 33,
                    ]
        self.channel_order = channel_order #maybe add if else loop for wave vs poly channels
        dr = data_reader(data_directory, stream, stimulus)
        self.signal_data, self.fs, self.stim_markers, self.animal_block = dr.get_data()
        self.marker_onsets = dr.get_stim_onsets()
        # self.tdt_data = tdt.read_block(data_directory)
        # self.fs = self.tdt_data['streams'][stream]['fs']
        # self.fs_stim_delay = .25 * self.fs
        # self.markers = self.tdt_data.epocs.mark.onset
        # self.marker_onsets = [int(x*self.fs+self.fs_stim_delay) for x in self.markers] 
        
        # wave_data_scrambled = self.tdt_data.streams.Wave.data
        # correct_shape_wave_data = wave_data_scrambled.T
        
        def channel_orderer(signal_data):
            """Puts the wave data into the order of the channels
            Args: 
            data: signal data in timepoints x channels
            chs (list): the correct order of the channels"""
            shape_wanted = signal_data.shape
            new_data = np.empty((shape_wanted[0], shape_wanted[1]))
    
            for i in np.arange(shape_wanted[1]):
                new_data[:, i] = signal_data[:, (self.channel_order[i] - 1)]
            return new_data
        
        self.new_signal_data = channel_orderer(self.signal_data)


    def get_trials_matrix(self, channel, pre_buf = 10000, post_buf = 10000):
        
        """Returns trial matrix
        Args:
            signal (np.array): wave data(for one channel) samples x channels
            markers (list): List of trial onset in samples
            pre_buf (int, optional): Number of samples to pull prior to baseline. Defaults to 10000.
            post_buf (int, optional): Number of samples to pull after. Defaults to 10000.
        Returns:
            trials_mat (np.array): Trial matrix samples by trials
        """
        
        nsamples = post_buf + pre_buf
        ntrials = len(self.marker_onsets)
        trials_mat = np.empty((nsamples, ntrials))
        channel_data = self.new_signal_data[:, channel]
        
        for idx, marker in enumerate(self.marker_onsets):
            start_frame, end_frame = marker - pre_buf, marker + post_buf
            trials_mat[:, idx] = channel_data[int(start_frame):int(end_frame)]
        return trials_mat

    
    
    
    def zscore_data(trials_matrix, num_base_pts=200):
        """Compute zscore across trial matrix
        Args:
            trials_matrix (nparray): Trial matrix of samples x trials
            num_base_pts (int, optional): The first num_base_pts are used for baseline. Defaults to 200.
        Returns:
                tm_norm_data (nparray): Normalized trial matrix
        """
        # Zscore the data
        mean = trials_matrix[:num_base_pts].mean(axis=0, keepdims=True)
        std = trials_matrix[:num_base_pts].std(axis=0, keepdims=True)
        tm_norm_data = (trials_matrix - mean) / std    
        return tm_norm_data
    
    
    
    
    def get_all_trials_matrices(self, pre_buf = 10000, post_buf = 10000):
        """
        Returns
        -------
        python dictionary where the key is the channel and the value is the trial matrix for that channel
        now, instead of calling get_trials_matrix a bunch of times when we want to visualize, we can 
        iterate over the keys in the all_trials matrix 

        """
        
        all_trials = {}
        for i in np.arange(len(self.channel_order)):
            one_channel = self.get_trials_matrix(i, pre_buf, post_buf)
            all_trials[self.channel_order[i]] = one_channel
        self.trials_dict = all_trials
        return all_trials  
    
    
    
    
    def compute_high_gamma(trials_mat, fs, pre_buf=10000, post_buf=10000, baseline=200):
        """Computes high gamma using wavelet transform
        Args:
            trials_mat (list): List of np-arrays that contain trial matrices (samples x trials)
            fs (numeric): Sample rate
            pre_buf (int, optional): The number of samples to include prior to sample midpoint of trials_mat. Defaults to 10000.
            post_buf (int, optional): The number of samples to include after the sample midpoint. Defaults to 10000.
            baseline (int, optiona): The first set of samples to include in baseline. Defaults to 200.
        Returns:
                high_gamma: Returns zscored wavelet magnitude coefficients at high gamma frequencies, samples x trials
        """
        Xh, _, f, _  = wavelet_transform(trials_mat, rate=fs, filters='rat', hg_only=True)
        n = Xh.shape[0] // 2
        Xh = Xh[(n - pre_buf):(n + post_buf), :, :] #throw away edges bc of possible effect affects
        Xm = abs(Xh) #take abs value
        Xnorm = viz.zscore_data(Xm, baseline) #zscore 
        high_gamma = Xnorm.mean(axis = -1)
        return high_gamma
    
    
    
    
    def plot_all_hg_response(self, channel, single_trials = False, fig = None, ax = None):
        """
        Plots the high gamma response for one channel 

        Parameters
        ----------
        self: viz object, must have trials dictionary
        channel: channel that you want to look at 
        single_trials: allows you to view the single trials, defaults to false
        single_trials : TYPE, optional
            DESCRIPTION. The default is True. Plots single trials, otherwise plots median +- std dev

        Returns
        -------
        Plot of high gamma response over time.

        """
        if not hasattr(self,'trials_dict'):
            raise AttributeError('Please run object.get_all_trial_matrices to build the trials matrix dictionary')
        trial_matrix = self.trials_dict[channel]
        high_gamma_data = viz.compute_high_gamma(trial_matrix, self.fs)
        num_trials = high_gamma_data.shape[1]
        x_axis = np.linspace(-10000, 10000, 20000)
        x_axis = (x_axis/self.fs) * 1000
    
        if fig == None and ax == None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig, ax
        fig.tight_layout()
        ax.axvline(0, ymin = -2, ymax = 500, color = 'darksalmon', zorder = 10000)
        median = np.median(high_gamma_data, axis = -1)
        ax.set_ylim(-5, max(median) + 15)
        ax.set_xlim(-200, 200)
        ax.plot(x_axis, median, color = 'k', zorder = 10)
        if single_trials:
            for i in np.arange(num_trials):
                ax.plot(x_axis, high_gamma_data[:, i], color = 'k', alpha = .05)
                ax.set_title("Channel {} High Gamma Response Single Trials".format(channel))
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Zscored High Gamma Response Coefficient")
        else:
            standard_dev = np.std(high_gamma_data, axis = -1)
            sqrt_n = np.sqrt(num_trials)
            standard_error = standard_dev/sqrt_n
            ax.fill_between(x_axis, median - standard_error, median + standard_error, color = 'k', alpha = .1)
            ax.set_title("Channel {} Median High Gamma Response".format(channel))
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Zscored High Gamma Response Coefficient")
        fig
    
        
    
    
    def plot_high_gamma_matrix(self, nrow, ncol):
        """Utilizes the trials dictionary to plot the high gamma responses from multiple channels in a matrix form
        Args:
            nrow (int): Number of rows
            ncol (int): Number of columns
        Returns:
            plots of the z-scored high gamma response coefficients over time
        """
        if not hasattr(self,'trials_dict'):
            raise AttributeError('Please run object.get_all_trial_matrices to build the trials matrix dictionary')
        chs = self.channel_order
    
        fig, axs = plt.subplots(nrow, ncol, figsize=(80, 40))
        fig.tight_layout()
        idx = 0 #starting point for index in grid 
        while idx < (nrow*ncol):
            row, col = idx // ncol, idx % ncol
            ax = axs[row, col]
            self.plot_all_hg_response(chs[idx], fig = fig, ax = ax)
            ax.set_title("Channel {}".format(chs[idx]))
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("High Gamma Response Coefficient")
            idx += 1
        animal_block = self.tdt_data.info.blockname
        fig.suptitle('High Gamma Response: {}'.format(animal_block), fontsize = 20, y = 1)
        fig
    
        
    
    
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
        Xnorm = viz.zscore_data(Xm, baseline) #zscore 
        return Xnorm, f
    
    
    
    
    def plot_spectrogram(spec_data, f, tmin, tmax, colorbar=False, ax=None, fig=None, zero_flag=False, log_scale=False):
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
        if (ax is None) or (fig is None):
            fig, ax = plt.subplots(1, 1)
        
        if log_scale:
            aspect = 10
        else:
            aspect = 1/5
        
        pos = ax.imshow(spec_data.T, interpolation='none', aspect=aspect, cmap='binary', 
                    origin='lower', extent=[tmin, tmax, f[0], f[-1]])
        if log_scale:
            ax.set_yscale('symlog', basey=2)
        if zero_flag:
            ax.plot([0 ,0], [f[0], f[-1]], 'r')
        if colorbar:
            fig.colorbar(pos, ax=ax, shrink=0.7, pad = 0.02)
            
    
    
    
    
    def plot_spectrogram_matrix(self, nrow, ncol, pre_buf=10000, post_buf=10000):
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
        chs = self.channel_order
        trials_dict = self.trials_dict
        idx = 0 #starting point for index in grid 
        while idx < (nrow*ncol):
            row, col = idx // ncol, idx % ncol
            #ch = chs[idx]
            ax = axs[row, col]
            channel = chs[idx]
            trials_mat = trials_dict[channel]
            tf_data, f = viz.compute_spectrogram(trials_mat, self.fs)
            tf_data = np.median(tf_data, axis=1)
            viz.plot_spectrogram(tf_data, f, -100, 100, ax=ax, fig=fig, colorbar=True,log_scale=False)
            ax.set_title("Channel {}".format(chs[idx]))
            ax.set_xlabel("Time (ms)")
#        ax.set_ylabel("Frequency (Hz)")
            idx += 1
        animal_block = self.tdt_data.info.blockname
        fig.suptitle('Neural Spectrogram: {}'.format(animal_block), fontsize = 20, y = 1)
        fig
        
      
        
      
        
    def plot_trials(self, channel, trials = True, mean = True): 
        '''Plot data trials and mean of trials per channel.
  
                Parameters
                ----------
                trial_list : (list)
                    list of np.arrays (trials), within an array (trial) is channels by samples
                stream : (str)
                    Name of stream in data block
                fs : TYPE
                    DESCRIPTION.
                trials : (plot, optional)
                    Whether to plot all trials. Defaults to True.
                mean : (plot, optional)
                    Whether to plot mean of all trials. Defaults to False.
            '''
        trial_list = self.trials_dict
        
        if isinstance(channel, list):
            
            if trials:
                for i in np.arange(len(channel)): 
                    plt.subplot(8, 16, i + 1)
    
                    for tidx, trial in enumerate(trial_list):
                        sub_trial = trial[channel[i],:20000]
                        plt.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
                        # ymin, ymax = np.min(sub_trial), np.max(sub_trial)
                        plt.axvline(x= 10025, ymin=0.05, ymax=0.95, color = 'darksalmon')
                        plt.xlim(9000, 11800)
                        #     plt.plot([.3*fs, .3*fs], [ymin, ymax], 'darksalmon')
                        plt.title('Ch. ' + str(channel[i]), fontsize= 9)
                        plt.xticks([9000, 10000, 11000],[-100, 0, 100])
                        plt.xticks(fontsize=10)
                        plt.yticks(fontsize=10)
                        plt.xlabel('time (ms)', fontsize= 8)
                        plt.ylabel('mV', fontsize= 8)
                    
                    
                    plt.suptitle('{} Average Trial Across Channels'.format(self.animal_block), fontsize=20, y=1)
            
            if mean:
                trial_mat = np.zeros((20000, len(trial_list)))
        
                for i in np.arange(len(channel)):
    
                    plt.subplot(8, 16, i + 1)
    
                    for tidx, trial in enumerate(trial_list):
                        sub_trial = trial[channel[i], :20000]
                        trial_mat[:, tidx] = sub_trial 
    
                    mean_trial = np.mean(trial_mat, axis=1)
                    plt.plot(mean_trial, color='k', linewidth=2.5, zorder=10)
                    plt.xlim(9000, 11800)
                    
        else:
            channel_matrix = trial_list[channel].T
            
            fig, ax = plt.subplots()
            fig.tight_layout()
            ax.set_xlim(9000, 11800)
            ax.set_xticks([9000, 10000, 11000])
            ax.set_xticklabels([-100, 0, 100])
            
            if trials:
                for tidx, trial in enumerate(channel_matrix):
                    sub_trial = trial[:]
                    # ymin, ymax = np.min(sub_trial), np.max(sub_trial)
                    ax.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
                    ax.axvline(x= 10025, ymin=0.05, ymax=0.95, color = 'darksalmon')
                
                ax.set_title("Channel {} Trials".format(channel))
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("μV")
            
            if mean:
                trial_mat = np.zeros((20000, len(channel_matrix)))
                
                for tidx, trial in enumerate(channel_matrix):
                    sub_trial = trial[:]
                    trial_mat[:, tidx] = sub_trial 
                    
                mean_trial = np.mean(trial_mat, axis=1)
                ax.plot(mean_trial, color='k', linewidth=2.5, zorder=10)
                
                ax.set_title("Channel {} Average Across Trials".format(channel))
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("μV")
        
        
        
        
        
        
        
        
        
        
        
        # stim_delay = 0.25
        # tdt_trials = tdt.epoc_filter(self.tdt_data, 'mark')
        # trial_list = tdt_trials.streams[self.stream].filtered
        # animal_block = self.tdt_data.info.blockname


        # if self.stream == "Wave":
        #     height = 8 
        #     width = 16
        #     tmax = 6000
        #     first, last = 2000, 5000
        #     chs = [
        #             81, 83, 85, 87, 89, 91, 93, 95, 97, 105, 98, 106, 114, 122, 113, 121,
        #             82, 84, 86, 88, 90, 92, 94, 96, 99, 107, 100, 108, 116, 124, 115, 123,
        #             66, 68, 70, 72, 74, 76, 78, 80, 101, 109, 102, 110, 118, 126, 117, 125,
        #             65, 67, 69, 71, 73, 75, 77, 79, 103, 111, 104, 112, 120, 128, 119, 127,
        #             63, 61, 59, 57, 55, 53, 51, 49, 25, 17, 26, 18, 10, 2, 9, 1,
        #             64, 62, 60, 58, 56, 54, 52, 50, 27, 19, 28, 20, 12, 4, 11, 3,
        #             48, 46, 44, 42, 40, 38, 36, 34, 29, 21, 30, 22, 14, 6, 13, 5,
        #             47, 45, 43, 41, 39, 37, 35, 33, 31, 23, 32, 24, 16, 8, 15, 7
        #             ]
        # if self.stream == "Poly":
        #     height = 32 
        #     width = 2
        #     tmax = 12000
        #     first, last = 5500, 7000
        #     chs = [ 
        #             27, 37,
        #             26, 38, 
        #             25, 39, 
        #             24, 40, 
        #             23, 41,
        #             22, 42, 
        #             21, 43, 
        #             20, 44, 
        #             19, 45, 
        #             18, 46,
        #             17, 47, 
        #             16, 48, 
        #             15, 49, 
        #             14, 50, 
        #             13, 51,
        #             12, 52, 
        #             11, 53, 
        #             10, 54, 
        #             9, 55, 
        #             8, 56, 
        #             7, 57, 
        #             6, 58, 
        #             5, 59, 
        #             4, 60, 
        #             3, 61, 
        #             2, 62, 
        #             1, 63, 
        #             28, 64, 
        #             29, 36, 
        #             30, 35, 
        #             31, 34, 
        #             32, 33,
        #             ]
          
        
        
        
        
