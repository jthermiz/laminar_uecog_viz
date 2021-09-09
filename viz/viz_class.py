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
import os
import warnings 
from process_nwb.wavelet_transform import wavelet_transform
from viz import data_reader as dr
import get_zscore 
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
        
        # def open_yaml_file(stimulus):
        #     with open('{stimulus_yamls}/{}.yaml'.format(stimulus), 'r') as file:
        #         stim_doc = yaml.full_load(file)
        #     return stim_doc
        
        if self.stream == "Wave" or "ECoG":
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
            channel_order = [ 
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
            
            # channel_order = [ 
            #         27, 37,
            #         26, 38, 
            #         25, 39, 
            #         24, 40, 
            #         23, 41,
            #         22, 42, 
            #         21, 43, 
            #         20, 44, 
            #         19, 45, 
            #         18, 46,
            #         17, 47, 
            #         16, 48, 
            #         15, 49, 
            #         14, 50, 
            #         13, 51,
            #         12, 52, 
            #         11, 53, 
            #         10, 54, 
            #         9, 55, 
            #         8, 56, 
            #         7, 57, 
            #         6, 58, 
            #         5, 59, 
            #         4, 60, 
            #         3, 61, 
            #         2, 62, 
            #         1, 63, 
            #         28, 64, 
            #         29, 36, 
            #         30, 35, 
            #         31, 34, 
            #         32, 33,
            #         ]
        self.channel_order = channel_order #maybe add if else loop for wave vs poly channels
        # self.height = height
        # self.width = width
        # self.first, self.last = first, last
        # self.figsize = figsize
        rd = dr.data_reader(data_directory, stream, stimulus)
        self.signal_data, self.fs, self.stim_markers, self.animal_block = rd.get_data()
        self.marker_onsets = rd.get_stim_onsets()
        
        self.savepic = '{}/Figures'.format(self.data_directory)
        if not os.path.exists(self.savepic):
            os.mkdir(self.savepic)
        if os.path.exists(self.savepic):
            print("Figure folder exists")
        
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

    
    
    
    def zscore_data(trials_mat, num_base_pts=600):
        """Compute zscore across trial matrix
        Args:
            trials_matrix (nparray): Trial matrix of samples x trials
            num_base_pts (int, optional): The first num_base_pts are used for baseline. Defaults to 200.
        Returns:
                tm_norm_data (nparray): Normalized trial matrix
        """
        # Zscore the data
        mean = trials_mat[:num_base_pts].mean(axis=0, keepdims=True)
        std = trials_mat[:num_base_pts].std(axis=0, keepdims=True)
        tm_norm_data = (trials_mat - mean) / std    
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
        # trials_dict = viz.get_all_trials_matrices(self)
        if not hasattr(self,'trials_dict'):
            raise AttributeError('Please run object.get_all_trial_matrices to build the trials matrix dictionary')
        trials_dict = viz.get_all_trials_matrices(self)
        trial_matrix = trials_dict[channel]
        high_gamma_data = viz.compute_high_gamma(trial_matrix, self.fs)
        num_trials = high_gamma_data.shape[1]
        x_axis = np.linspace(-10000, 10000, 20000)
        #x_axis = np.linspace(-10000, 15500, 20000) #wave tone_diag
        #x_axis = np.linspace(-10000, 5500, 20000)
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
        ax.plot(x_axis, median, color = 'k', linewidth= 2, zorder = 10)
        if single_trials:
            for i in np.arange(num_trials):
                ax.plot(x_axis, high_gamma_data[:, i], color = 'k', alpha = .05)
                ax.set_title("Channel {} High Gamma Response Single Trials".format(channel))
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Zscored High Gamma Response")
        else:
            standard_dev = np.std(high_gamma_data, axis = -1)
            sqrt_n = np.sqrt(num_trials)
            standard_error = standard_dev/sqrt_n
            ax.fill_between(x_axis, median - standard_error, median + standard_error, color = 'k', alpha = .15)
            ax.set_title("Channel {} Median High Gamma Response".format(channel))
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Zscored High Gamma Response")
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
    
        fig, axs = plt.subplots(nrow, ncol, figsize=(20, 15))
        fig.tight_layout(rect = (0.01, 0, 1, 0.97))
        idx = 0 #starting point for index in grid 
        while idx < (nrow*ncol):
            row, col = idx // ncol, idx % ncol
            ax = axs[row, col]
            self.plot_all_hg_response(chs[idx], fig = fig, ax = ax)
            ax.set_title("Channel {}".format(chs[idx]))
            ax.set_xlabel("Time (ms)")
            #ax.set_ylabel("High Gamma Response Z-Scored")
            idx += 1
        animal_block = self.animal_block
        fig.suptitle('High Gamma Response: {}'.format(animal_block), fontsize = 20, y = 1)
        fig.supylabel('High Gamma Response Z-Scored', fontsize = 15)
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
        Xh = Xh[:, :, :] #throw away low frequencies
        f = f[:]
        Xm = abs(Xh) #take abs value
        Xnorm = viz.zscore_data(Xm, baseline) #zscore 
        print('f[0]', f[0])
        print('f[-1]', f[-1])
        return Xnorm, f
    
    
    
    
    def plot_spectrogram(tf_data, f, tmin, tmax, colorbar=False, ax=None, fig=None, zero_flag=False, vrange=[0, None], max_flag=True):
        """Plots spectrogram
        spec_data, f, tmin, tmax, colorbar=False, ax=None, fig=None, zero_flag=False, log_scale=False
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
            
        mat = tf_data.T
        print(mat.shape)
        
        pos = ax.imshow(mat, interpolation='none', aspect=2, vmin=vrange[0], vmax=vrange[1], cmap='binary', origin='lower', extent=[tmin, tmax, 0, len(f)])
        
        values = [10, 30, 75, 150, 300, 600, 1200]
        #values = [3, 7, 24, 75, 240, 760]
        
        positions = [np.argmin(np.abs(v - f)) for v in values]
        
        plt.yticks(positions, values)
        #ax.set_yticklabels(values)
        #print(positions)
        
        
        if zero_flag:
            ax.plot([0 ,0], [0, len(f)], 'r')
        if colorbar:
            fig.colorbar(pos, ax=ax)
        if max_flag:
        #textstr = r'$Z_{min}=%.2f$' % (np.max(tf_data))
            textstr = 'min={0:.2f}, max={1:.2f}'.format(np.min(tf_data), np.max(tf_data))
            ax.set_title(textstr, fontsize=8)
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
        
        return fig, ax 
            
         
            
         
            
        # if (ax is None) or (fig is None):
        #     fig, ax = plt.subplots(1, 1)
        
        # if log_scale:
        #     aspect = 10
        # else:
        #     aspect = 1/5
        
        # pos = ax.imshow(spec_data.T, interpolation='none', aspect=aspect, cmap='binary', 
        #             origin='lower', extent=[tmin, tmax, f[0], f[-1]], vmin=0)
        # if log_scale:
        #     ax.set_yscale('symlog', basey=2)
        # if zero_flag:
        #     ax.plot([0 ,0], [f[0], f[-1]], 'r')
        # if colorbar:
        #     fig.colorbar(pos, ax=ax, shrink=0.7, pad = 0.02)
            
    
    
    
    
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
        if self.stream == "Wave" or "ECoG":
            figsize = (60, 30)
            layout = [0.035, 0.03, 0.95, 0.97]
        if self.stream == "Poly":
            figsize = (8, 80)
            layout = [0.09, 0.015, 0.97, 0.992]
        
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
        fig.tight_layout(rect = layout)
        chs = self.channel_order
        trials_dict = viz.get_all_trials_matrices(self)
        idx = 0 #starting point for index in grid 
        while idx < (nrow*ncol):
            row, col = idx // ncol, idx % ncol
            #ch = chs[idx]
            ax = axs[row, col]
            channel = chs[idx]
            trials_mat = trials_dict[channel]
            tf_data, f = viz.compute_spectrogram(trials_mat, self.fs)
            tf_data = np.median(tf_data, axis=1)
            viz.plot_spectrogram(tf_data, f, -100, 100, ax=ax, fig=fig, colorbar=True,zero_flag=False)
            ax.set_title("Channel {}".format(chs[idx]))
            ax.set_xlabel("Time (ms)")
#        ax.set_ylabel("Frequency (Hz)")
            idx += 1
        animal_block = self.animal_block
        fig.suptitle('Neural Spectrogram: {}'.format(animal_block), fontsize = 30, y = 1)
        fig
        
      
    def plot_zscore(self, channel, fig = None, ax = None, labels = True):
        trials_dict = viz.get_all_trials_matrices(self)
        trials_mat = trials_dict[channel]
            
        if fig == None and ax == None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig, ax    
        fig.tight_layout(rect = (0.04, 0.04, 1, 0.95))
            
        x_axis = np.linspace(-10000, 10000, 20000)
        #x_axis = np.linspace(-10000, 7500, 20000) #wave tonediag
        #x_axis = np.linspace(-10000, 5500, 20000)  #poly tonediag
        x_axis = (x_axis/self.fs) * 1000

        ax.set_xlim(-150, 150)
        #ax.set_xlim(-150, 150)
        
        data_for_channel_zscored = viz.zscore_data(trials_mat)
        average_for_channel = np.mean(data_for_channel_zscored, axis = 1)
            
        # onset_start = int(.05*self.fs)
        # onset_stop = int(.15*self.fs)
                
        # data_for_channel_zscored = get_zscore.zscore_from_baseline(trials_dict, channel, onset_start, onset_stop)
        # average_for_channel = get_zscore.get_average_zscore(trials_dict, channel, onset_start, onset_stop)
        
        ax.set_ylim(min(average_for_channel)-1, max(average_for_channel)+2)    
        
        
        zscored_data = data_for_channel_zscored.T
        
        print(zscored_data.shape)
        print(len(zscored_data))
            
        trial_mat = np.zeros((20000, len(zscored_data)))
                
        for tidx, trial in enumerate(zscored_data):
            sub_trial = trial[:]
            trial_mat[:, tidx] = sub_trial
            ax.plot(x_axis, sub_trial, color = 'grey', alpha = .035, linewidth=0.5)
                
        num_trials = data_for_channel_zscored.shape[1]
        mean = np.mean(data_for_channel_zscored, axis = -1)
                
        standard_dev = np.std(data_for_channel_zscored, axis = -1)
        sqrt_n = np.sqrt(num_trials)
        standard_error = standard_dev/sqrt_n
        ax.fill_between(x_axis, mean - standard_error, mean + standard_error, color = 'k', alpha = .3)
                    
        ax.plot(x_axis, average_for_channel, color = 'k', linewidth= 2, zorder = 9)
        ax.axvline(x= 0, ymin=min(zscored_data.flatten()), ymax=max(zscored_data.flatten()), color = 'darksalmon', zorder = 11)
                
        ax.set_title("Channel {} Average Zscored Trial".format(channel), fontsize = 15)
        
        if labels == True:
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("mV")
        
                
        # fig.savefig("{}/{}_Zscore_Ch.{}_{}.png".format(self.savepic, self.animal_block, channel, self.stream), dpi=300)
        
    def plot_zscore_matrix(self, nrow, ncol, selected_chs = None):
        
        #trials_dict = viz.get_all_trials_matrices(self)
        
        if isinstance(selected_chs, list):
            chs = selected_chs
            figsize = (10, 9)
            titlesize = 18
            labelsize = 12
        else: 
            chs = self.channel_order
            figsize = (40, 40)
            titlesize = 50
            labelsize = 30
        
        # if self.stream == "Wave" or "ECoG":
        #     figsize = (60, 30)
        #     layout = [0.035, 0.03, 0.95, 0.97]
        #     titlesize = 55
        #     labelsize = 45
        # if self.stream == "Poly":
        #     figsize = (8, 80)
        #     layout = [0.09, 0.015, 0.97, 0.992]
        #     titlesize = 30
        #     labelsize = 20
            
            
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
        fig.tight_layout()
        idx = 0 #starting point for index in grid
            
        
        # if self.stream == "Wave" or "ECoG":
        #     fig.supxlabel('Time (ms)', fontsize = labelsize)
        #     fig.suptitle('{} Zscore Response Across Channels'.format(self.animal_block), fontsize = titlesize, y = 1)
        #     fig.supylabel('mV', fontsize = labelsize)
        # if self.stream == "Poly":
        #     fig.supylabel('mV', fontsize = labelsize)
        #     fig.supxlabel('Time (ms)', fontsize = labelsize)
        #     fig.suptitle('{} Zscore Response Across Channels'.format(self.animal_block), fontsize = titlesize, y = 1)
       
                
        while idx < (nrow*ncol):
            row, col = idx // ncol, idx % ncol
            ax = axs[row, col]
            self.plot_zscore(chs[idx], fig = fig, ax = ax, labels = False)
            idx += 1
        
        animal_block = self.animal_block
        fig.suptitle('Z-Scored Responses Across Channels: {}'.format(animal_block), fontsize = titlesize, y = 1)
        fig.supylabel('mV', fontsize = labelsize)
        fig.supxlabel('Time(ms)', fontsize = labelsize)
         
        
    def plot_trials(self, channel, trials = True, zscore = False): 
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
                    Whether to plot all trials & mean of all trials. Defaults to True.
            '''
        trials_dict = viz.get_all_trials_matrices(self)
        #savepic = self.savepic
        
        if self.stream == "Wave" or "ECoG":
            height = 8 
            width = 16
            #tmax = 6000
            first, last = 9000, 11800
            figsize = (60, 30)
            layout = [0.035, 0.03, 0.95, 0.97]
            stim = .825
        if self.stream == "Poly":
            height = 32 
            width = 2
            #tmax = 12000
            first, last = 9000, 11800
            figsize = (8, 80)
            layout = [0.09, 0.015, 0.97, 0.992]
            stim = .4125
        
        if isinstance(channel, list):
            
            fig, axs = plt.subplots(height, width, figsize=figsize)
            fig.tight_layout(rect = layout)
            if self.stream == "Wave" or "ECoG":
                fig.supylabel('μV', fontsize = 45)
                fig.supxlabel('Time (ms)', fontsize = 45)
                if trials:
                    fig.suptitle('{} Average Trial Across Channels'.format(self.animal_block), fontsize = 55, y = 1)
                if zscore:
                    fig.suptitle('{} Zscore Response Across Channels'.format(self.animal_block), fontsize = 55, y = 1)
                    fig.supylabel('mV', fontsize = 45)
            if self.stream == "Poly":
                fig.supylabel('mV', fontsize = 20)
                fig.supxlabel('Time (ms)', fontsize = 20)
                if trials:
                    fig.suptitle('{} Average Trial Across Channels'.format(self.animal_block), fontsize = 30, y = 1)
                if zscore:
                    fig.suptitle('{} Zscore Response Across Channels'.format(self.animal_block), fontsize = 30, y = 1)
            chs = self.channel_order
            idx = 0 #starting point for index in grid 
            if trials:
                while idx < (height*width):
                    row, col = idx // width, idx % width
                    #ch = chs[idx]
                    ax = axs[row, col]
                    channel = chs[idx]
                    trials_matrix = trials_dict[channel].T
                    trial_mat = np.zeros((20000, len(trials_matrix)))
                    ax.set_xlim(first, last)
                    ax.set_xticks([first, first + 1000, last - 800])
                    ax.set_xticklabels([-100, 0, 100])
                    for tidx, trial in enumerate(trials_matrix):
                        sub_trial = trial[:]
                        trial_mat[:, tidx] = sub_trial 
                        ax.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
                        ax.axvline(x= stim*self.fs, ymin=0.05, ymax=0.95, color = 'darksalmon')
                    mean_trial = np.mean(trial_mat, axis=1)
                    ax.plot(mean_trial, color='k', linewidth=2.5, zorder=10)
                        
                    ax.set_title("Channel {}".format(chs[idx]))
                    idx += 1
                    
                fig.savefig("{}/{}_Average_Trials_Across_Channels_{}.png".format(self.savepic, self.animal_block, self.stream), dpi=300)
                    
            if zscore:
                onset_start = int(.05*self.fs)
                onset_stop = int(.15*self.fs)
                
                while idx < (height*width):
                    row, col = idx // width, idx % width
                    #ch = chs[idx]
                    ax = axs[row, col]
                    channel = chs[idx]
                    ax.set_xlim(first, last)
                    ax.set_xticks([first, first + 1000, last - 800])
                    ax.set_xticklabels([-100, 0, 100])
                    
                    data_for_channel_zscored = get_zscore.zscore_from_baseline(trials_dict, channel, onset_start, onset_stop)
                    average_for_channel = get_zscore.get_average_zscore(trials_dict, channel, onset_start, onset_stop)
                    std_for_channel = get_zscore.get_std(trials_dict, channel, onset_start, onset_stop)
                        
                    ax.plot(average_for_channel, color = 'k', linewidth= 2.5, zorder = 9)
                    ax.plot(average_for_channel + std_for_channel, 'w', average_for_channel - std_for_channel, 'w', linewidth=.8, zorder=10)
                    ax.fill_between(range(20000), average_for_channel - std_for_channel, average_for_channel + std_for_channel, color='gray', alpha = 0.5)
                    ax.vlines(stim*self.fs, ymin = min(data_for_channel_zscored.flatten()), ymax = max(data_for_channel_zscored.flatten()), color = 'darksalmon', zorder = 11)
                    # ax.axhline(y=0, color='darkgrey', linestyle='--', zorder = 1)
                    ax.set_title("Channel {}".format(chs[idx]))
                    idx += 1
                
                fig.savefig("{}/{}_ZScore_Across_Channels_{}.png".format(self.savepic, self.animal_block, self.stream), dpi=300)
                    
            
                    
        else:
            channel_matrix = trials_dict[channel].T
            # print(channel_matrix.shape)
            
            fig, ax = plt.subplots()
            fig.tight_layout()
            ax.set_xlim(first, last)
            ax.set_xticks([first, first + 1000, last - 800])
            ax.set_xticklabels([-100, 0, 100])
            
            
            if trials:
                trial_mat = np.zeros((20000, len(channel_matrix)))
                
                for tidx, trial in enumerate(channel_matrix):
                    sub_trial = trial[:]
                    trial_mat[:, tidx] = sub_trial
                    ax.plot(sub_trial, color=(.85,.85,.85), linewidth=0.5)
                    ax.axvline(first + 1025, ymin=0.05, ymax=0.95, color = 'darksalmon')
                    
                mean_trial = np.mean(trial_mat, axis=1)
                ax.plot(mean_trial, color='k', linewidth=2.5, zorder=10)
                
                ax.set_title("Channel {} Average Across Trials".format(channel))
                ax.set_xlabel("Time (ms)")
                if self.stream == "Wave" or "ECoG":
                    ax.set_ylabel("μV")
                if self.stream == "Poly":
                    ax.set_ylabel("mV")
                    
                fig.savefig("{}/{}_Average_Trial_Ch.{}_{}.png".format(self.savepic, self.animal_block, channel, self.stream), dpi=300)
                    
            if zscore:
                onset_start = int(.05*self.fs)
                onset_stop = int(.15*self.fs)
                
                data_for_channel_zscored = get_zscore.zscore_from_baseline(trials_dict, channel, onset_start, onset_stop)
                average_for_channel = get_zscore.get_average_zscore(trials_dict, channel, onset_start, onset_stop)
                std_for_channel = get_zscore.get_std(trials_dict, channel, onset_start, onset_stop)
                
                print("data for channel", data_for_channel_zscored.shape)
                zscored_data = data_for_channel_zscored.T
                num_trials = zscored_data.shape[1]
                median = np.median(zscored_data, axis = -1)
                
                standard_dev = np.std(zscored_data, axis = -1)
                sqrt_n = np.sqrt(num_trials)
                standard_error = standard_dev/sqrt_n
                ax.fill_between(range(20000), median - standard_error, median + standard_error, color = 'k', alpha = .2)
                    
                ax.plot(average_for_channel, color = 'k', linewidth= 2.5, zorder = 9)
                #ax.plot(average_for_channel + std_for_channel, 'w', average_for_channel - std_for_channel, 'w', linewidth=.8, zorder=10)
                #ax.fill_between(range(20000), average_for_channel - std_for_channel, average_for_channel + std_for_channel, color='gray', alpha = 0.5)
                # ax.axhline(y=0, color='darkgrey', linestyle='--', zorder = 1)
                ax.vlines(stim*self.fs, ymin = min(data_for_channel_zscored.flatten()), ymax = max(data_for_channel_zscored.flatten()), color = 'darksalmon', zorder = 11)
                #ax.vlines(.4425*self.fs, ymin = min(data_for_channel_zscored.flatten()), ymax = max(data_for_channel_zscored.flatten()), color = 'darksalmon', zorder = 11)
                
                ax.set_title("Channel {} Average Zscored Trial".format(channel))
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("mV")
                
                fig.savefig("{}/{}_Zscore_Ch.{}_{}.png".format(self.savepic, self.animal_block, channel, self.stream), dpi=300)
                
                
            
            # if mean:
            #     trial_mat = np.zeros((20000, len(channel_matrix)))
                
            #     for tidx, trial in enumerate(channel_matrix):
            #         sub_trial = trial[:]
            #         trial_mat[:, tidx] = sub_trial 
                    
            #     mean_trial = np.mean(trial_mat, axis=1)
            #     ax.plot(mean_trial, color='k', linewidth=2.5, zorder=10)
                
            #     ax.set_title("Channel {} Average Across Trials".format(channel))
            #     ax.set_xlabel("Time (ms)")
            #     ax.set_ylabel("μV")
        
        
        
        
        
        
        
        
        
        
        
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
          
        
        
        
        
