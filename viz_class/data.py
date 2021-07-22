#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 09:38:27 2021

@author: vanessagutierrez
"""

import numpy as np 
import tdt
import os
import warnings
import yaml

class data:
    
    def __init__(self, data_directory, stream, stimulus):
        self.data_directory = data_directory
        self.stream = stream
        self.stimulus = stimulus
        self.tdt_data = tdt.read_block(data_directory)
        self.fs = self.tdt_data['streams'][stream]['fs']
        self.fs_stim_delay = .25 * self.fs
        
        def open_yaml(stimulus):
            with open('stimulus_yamls/{}.yaml'.format(stimulus), 'r') as file:
                stim_doc = yaml.full_load(file)
            return stim_doc
        
        self.stim_doc = open_yaml(stimulus)
    
    def get_data(self):
        '''
        Read data block, extracts stream data, extracts sample rate, extracts data trials.
        Parameters
        ----------
        data_directory : (path)
            Data path
        stream : (str)
            Name of stream in data block
        epoc_event : (str)
            Name of epoc in data block
        Returns
        -------
        stream_data : (np.array)
            array of channels by samples
        fs : (np.float)
            sample rate
        trial_list : (list)
            list of np.arrays (trials), within an array (trial) is channels by samples
        animal_block : (str)
            name of animal block
        '''
        if self.data_directory.endswith('.nwb'):
            io = NWBHDF5IO(data_directory, 'r')
            nwbfile_in = io.read()
            signal_data = nwbfile_in.acquisition[stream].data
            fs = nwbfile_in.acquisition[stream].rate
            markers = nwbfile_in.stimulus['recorded_mark'].data
        
        
        
        tdt_data = tdt.read_block(data_directory)
        animal_block = tdt_data.info.blockname
        stream_data = tdt_data.streams[stream].data
        fs = tdt_data.streams[stream].fs
        #tdt_trials = tdt.epoc_filter(tdt_data, 'mark')
        #trial_list = tdt_trials.streams[stream].filtered
        
        return stream_data, fs, trial_list, animal_block
    
    def open_yaml(stimulus):
        with open('stimulus_yamls/{}.yaml'.format(stimulus), 'r') as file:
            stim_doc = yaml.full_load(file)
        return stim_doc


    def get_stim_onsets(nwbfile, stimulus):
        stim_doc = open_yaml(stimulus)
    
        for item, doc in documents.items():
            print(item, ":", doc)
        mark_data = nwbfile.stimulus['recorded_mark'].data
        mark_fs = nwbfile.stimulus['recorded_mark'].rate
        mark_offset = nsenwb.stim['mark_offset']
        stim_dur = nsenwb.stim['duration']
        stim_dur_samp = stim_dur*mark_fs
    
        mark_threshold = 0.25 if nsenwb.stim.get('mark_is_stim') else nsenwb.stim['mark_threshold']
        thresh_crossings = np.diff( (mark_dset.data[:] > mark_threshold).astype('int'), axis=0 )
        stim_onsets = np.where(thresh_crossings > 0.5)[0] + 1 # +1 b/c diff gets rid of 1st datapoint
    
        real_stim_onsets = [stim_onsets[0]]
        for stim_onset in stim_onsets[1:]:
            # Check that each stim onset is more than 2x the stimulus duration since the previous
            if stim_onset > real_stim_onsets[-1] + 2*stim_dur_samp:
                real_stim_onsets.append(stim_onset)
    
        if len(real_stim_onsets) != nsenwb.stim['nsamples']:
            print("WARNING: found {} stim onsets in block {}, but supposed to have {} samples".format(
                len(real_stim_onsets), nsenwb.block_name, nsenwb.stim['nsamples']))
            
        stimulus_onsets = (real_stim_onsets / mark_fs)
            
        return stimulus_onsets

