#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:15:14 2021

@author: vanessagutierrez
"""

import data_reader as dr

data_directory = r'/Users/vanessagutierrez/data/Rat/RVG14/RVG14_B01'
stream = 'Wave'
stimulus = 'wn2' #change str format on class so that it reads wn2 instead of 'wn2'

dr_test = dr.data_reader(data_directory, stream, stimulus)
dr.data_reader(data_directory, stream, stimulus)

signal_data, fs, stim_markers, animal_block = dr_test.get_data()
signal_data

marker_onsets = dr_test.get_stim_onsets()
marker_onsets()
marker_onsets
len(marker_onsets)

#look into parameters for each class and how to keep the ones for reader to reader and viz to viz