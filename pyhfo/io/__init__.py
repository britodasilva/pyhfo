# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:04:40 2015

@author: anderson
"""

import h5py
import numpy as np


def open_dataset(file_name,dataset_name):
    '''
    open a dataset in a specific file_name, return Data_dict
    '''
    # reading h5 file
    h5 = h5py.File(file_name,'r')
    # loading dataset
    dataset = h5[dataset_name]
    # storing Sample Rate attribute
    sample_rate = dataset.attrs['SampleRate[Hz]']
    n_points         = dataset.shape[0]
    end_time         = n_points/sample_rate
    # Time vector
    time_vec         = np.linspace(0,end_time,n_points,endpoint=False)
    # Creating dictionary
    Data_dict = {"raw_signal": dataset[:], "sample_rate": sample_rate, "n_channels": dataset.shape[1], "ch_labels": dataset.attrs['Channel_Labels'], "time_vec": time_vec} 
    return Data_dict
    
    
    