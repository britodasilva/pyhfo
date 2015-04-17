# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:12:23 2015

@author: anderson
"""

import h5py
import numpy as np


def open_dataset(file_name,dataset_name):
    '''
    open a dataset in a specific file_name, return Data_dict
    '''
    # reading h5 file
    h5 = h5py.File(file_name,'r+')
    # loading dataset
    dataset = h5[dataset_name]
    # Sample Rate attribute
    sample_rate = dataset.attrs['SampleRate[Hz]']
    n_points         = dataset.shape[0]
    end_time         = n_points/sample_rate
    # Time vector
    time_vec         = np.linspace(0,end_time,n_points,endpoint=False)
    # Check if has 'Bad_channels' attribute, if not, create one empty
    if len([x for x in dataset.attrs.keys() if x == 'Bad_channels']) == 0:
        dataset.attrs.create("Bad_channels",[],dtype=int)
    # Load bad channels
    bad_channels = dataset.attrs["Bad_channels"]    
    # Creating dictionary
    Data_dict = {"data": dataset[:], "sample_rate": sample_rate, "n_channels": dataset.shape[1], "ch_labels": dataset.attrs['Channel_Labels'], "time_vec": time_vec, "bad_channels": bad_channels} 
    h5.close()
    return Data_dict