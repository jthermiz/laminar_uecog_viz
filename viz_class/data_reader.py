#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 09:38:27 2021

@author: vanessagutierrez

Class to read raw data from either NWB files or TDT into an objects that can be easily used for data analysis.
"""

import tdt
import yaml
from pynwb import NWBHDF5IO

class data_reader:
    
    def __init__(self, data_directory, stream, stimulus):
        '''
        Init: create an instance of the data_reader class

        Parameters
        ----------
        data_directory : PATH
            Data path.
        stream : STR
            Name of stream in data block (ex: 'Wave' or 'ECoG').
        stimulus : STR
            Name of stimulus in data block (ex: 'wn2' or 'tone_diagnostic').

        Returns
        -------
        stim_doc : DICT
            Dictionary of yaml file contaning relevant stimulus values

        '''
        self.data_directory = data_directory
        self.stream = stream
        self.stimulus = stimulus
        
        def open_yaml(self):
            with open('stimulus_yamls/{}.yaml'.format(self.stimulus), 'r') as file:
                stim_doc = yaml.full_load(file)
            return stim_doc
        
        self.stim_doc = open_yaml(stimulus)
        
    
    def get_data(self):
        '''
        Read data block, extracts stream data, sample rate, stimulus markers, and animal block id.

        Returns
        -------
        signal_data: (np.array)
            array of samples by channels
        fs: (np.float)
            sample rate
        stim_markers : (list)
            list of stimulus markers
        animal_block : (str)
            name of animal block

        '''
        if self.data_directory.endswith('.nwb'):
            io = NWBHDF5IO(self.data_directory, 'r')
            nwbfile_in = io.read()
            self.animal_block = nwbfile_in.session_id
            self.signal_data = nwbfile_in.acquisition[self.stream].data
            self.fs = nwbfile_in.acquisition[self.stream].rate
            markers = nwbfile_in.trials.to_dataframe()
            self.stim_markers = markers[markers["sb"] == "s"]
        
        else:
            tdt_data = tdt.read_block(self.data_directory)
            self.animal_block = tdt_data.info.blockname
            stream_data = tdt_data.streams[self.stream].data
            self.fs = tdt_data.streams[self.stream].fs
            self.signal_data = stream_data.T
            self.stim_markers = self.tdt_data.epocs.mark
    
        
        return self.signal_data, self.fs, self.stim_markers, self.animal_block
    

    def get_stim_onsets(self):
        '''
        
        Returns
        -------
        marker_onsets : LIST
            DESCRIPTION.

        '''
        mark_offset = self.stim_doc['mark_offset']
        fs_stim_delay = self.fs * mark_offset
        nsamples = self.stim_doc['nsamples']
        
        if self.data_directory.endswith('.nwb'):
            onsets = self.stim_markers.iloc[:, [0,2]]
            markers = onsets['start_time'].to_list()
            marker_onsets = [int(x*self.fs+fs_stim_delay) for x in markers]
            
        else:
            markers = self.stim_markers.onset
            marker_onsets = [int(x*self.fs+fs_stim_delay) for x in markers]
            
        if len(marker_onsets) != nsamples:
            print("WARNING: found {} stim onsets in block {}, but supposed to have {} samples".format(len(marker_onsets), self.animal_block, nsamples))
        
        
        return marker_onsets





# mark_data = nwbfile.stimulus['recorded_mark'].data
#         mark_fs = nwbfile.stimulus['recorded_mark'].rate
#         mark_offset = nsenwb.stim['mark_offset']
#         stim_dur = nsenwb.stim['duration']
#         stim_dur_samp = stim_dur*mark_fs
    
#         mark_threshold = 0.25 if nsenwb.stim.get('mark_is_stim') else nsenwb.stim['mark_threshold']
#         thresh_crossings = np.diff( (mark_dset.data[:] > mark_threshold).astype('int'), axis=0 )
#         stim_onsets = np.where(thresh_crossings > 0.5)[0] + 1 # +1 b/c diff gets rid of 1st datapoint
    
#         real_stim_onsets = [stim_onsets[0]]
#         for stim_onset in stim_onsets[1:]:
#             # Check that each stim onset is more than 2x the stimulus duration since the previous
#             if stim_onset > real_stim_onsets[-1] + 2*stim_dur_samp:
#                 real_stim_onsets.append(stim_onset)