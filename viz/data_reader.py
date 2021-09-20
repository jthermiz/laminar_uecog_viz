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
        data_directory (path): Data path.
        stream (str): Name of stream in data block (ex: 'Wave' or 'ECoG').
        stimulus (str): Name of stimulus in data block (ex: 'wn2' or 'tone_diagnostic').

        Returns
        -------
        stim_doc (dict): Dictionary of yaml file contaning relevant stimulus values

        '''
        self.data_directory = data_directory
        self.stream = stream
        self.stimulus = stimulus
        
        def open_yaml(stimulus):
            with open('stimulus_yamls/{}.yaml'.format(stimulus), 'r') as file:
                stim_doc = yaml.full_load(file)
            return stim_doc
        
        self.stim_doc = open_yaml(stimulus)
        
    
    def get_data(self):
        '''
        Read data block, extracts stream data, sample rate, stimulus markers, and animal block id.

        Returns
        -------
        signal_data (np.array): array of samples by channels
        fs (np.float): sample rate
        stim_markers (list): list of stimulus markers
        animal_block (str): name of animal block

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
            #self.data_directory = ‘r’ + data_directory
            tdt_data = tdt.read_block(self.data_directory)
            self.animal_block = tdt_data.info.blockname
            stream_data = tdt_data.streams[self.stream].data
            self.fs = tdt_data.streams[self.stream].fs
            self.signal_data = stream_data.T
            self.stim_markers = tdt_data.epocs.mark
    
        
        return self.signal_data, self.fs, self.stim_markers, self.animal_block
    

    def get_stim_onsets(self):
        '''
        
        Returns
        -------
        marker_onsets (list): marker onset timepoints in samples.
        stim_duration (int): duration of stimulus in seconds
        
        '''
        mark_offset = self.stim_doc['mark_offset']
        fs_stim_delay = self.fs * mark_offset
        nsamples = self.stim_doc['nsamples']
        stim_duration = self.stim_doc['duration']
        
        if self.data_directory.endswith('.nwb'):
            onsets = self.stim_markers.iloc[:, [0,2]]
            markers = onsets['start_time'].to_list()
            marker_onsets = [int(x*self.fs+fs_stim_delay) for x in markers]
            
        else:
            markers = self.stim_markers.onset
            marker_onsets = [int(x*self.fs+fs_stim_delay) for x in markers]
            
        if len(marker_onsets) != nsamples:
            print("WARNING: found {} stim onsets in block {}, but supposed to have {} samples".format(len(marker_onsets), self.animal_block, nsamples))
        
        
        return marker_onsets, stim_duration
