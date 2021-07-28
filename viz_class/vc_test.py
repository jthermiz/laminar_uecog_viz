#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:34:23 2021

@author: macproizzy
"""

import viz_class as vc

RVG06_B03 = r'/Users/vanessagutierrez/data/Rat/RVG06/RVG06_B03'
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

test = vc.viz(RVG06_B03, chs_ordered, stream)
test.get_all_trials_matrices()

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




