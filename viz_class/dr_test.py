#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:15:14 2021

@author: vanessagutierrez
"""

import data_reader as dr

data_directory = r'/Users/vanessagutierrez/data/Rat/RVG14/RVG14_B01'
stream = 'Wave'
stimulus = 'wn2'

dr_test = dr.data_reader(data_directory, stream, stimulus)


signal_data, fs, stim_markers, animal_block 