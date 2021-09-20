#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:54:45 2021

@author: vanessagutierrez
"""
import numpy as np 

def channel_orderer(signal_data, channel_order):
    """Puts the wave data into the order of the channels
    Args: 
    data: signal data in timepoints x channels
    chs (list): the correct order of the channels"""
    shape_wanted = signal_data.shape
    new_data = np.empty((shape_wanted[0], shape_wanted[1]))
    
    for i in np.arange(shape_wanted[1]):
        new_data[:, i] = signal_data[:, (channel_order[i] - 1)]
    return new_data