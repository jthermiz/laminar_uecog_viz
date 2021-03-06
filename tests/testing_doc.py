#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:00:38 2021

@author: macproizzy

Description: Testing Script 

- imports different scripts, generates plots of each type 
"""

import just_high_gamma as hg
import just_spectrogram_matrix as spec

data_directory = r'/Users/macproizzy/Desktop/Raw_Signal/RVG02_B01'
stream = 'Wave'
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


new_wave_data, marker_onsets, fs = spec.get_data(data_directory, stream)
unscrambled = spec.channel_orderer(new_wave_data, chs_ordered)
spec.plot_spectrogram_matrix(unscrambled, fs, marker_onsets, chs_ordered, nrow = 2, ncol = 16)



hg.plot_high_gamma(data_directory, 15, 25)
hg.plot_heatmap(data_directory, 15)

