# -*- coding: utf-8 -*-
"""
Pre-processing using Data_dict
Created on Fri Apr 17 13:15:57 2015
@author: anderson
"""

import scipy.signal as sig
import numpy as np

def decimate(Data_dict,q):
    '''
    Use scipy decimate to create a new Data_dict with low sample rate data
    '''
    #reading data
    data = Data_dict['data']
    # get shape
    npoints, nch = data.shape
    # creating a empty array
    new_data = np.empty((npoints/q,nch))
    new_data[:] = np.NAN
    # decimate each channel
    for ch in range(nch):
        new_data[:,ch] = sig.decimate(data[:,ch],q)
    # calculate new sample rate    
    new_sample_rate = Data_dict["sample_rate"]/q
    # creating new time_vec
    new_time_vec = Data_dict["time_vec"][0:-1:q]
    # creating new Data_dict
    new_Data_dict = {"data": new_data, "sample_rate": new_sample_rate, 
                     "n_channels": nch, "ch_labels": Data_dict["ch_labels"], 
                     "time_vec": new_time_vec, "bad_channels": Data_dict["bad_channels"]} 
    return new_Data_dict
    
def resample(Data_dict,q):
    '''
    Slice data with quocient q. sample_rate % q should be 0
    '''
    if Data_dict["sample_rate"] % q != 0:
        print 'sample_rate % q should be int'
        return
    else:
        #reading data
        data = Data_dict['data']
        new_data = data[0:-1:q,:]
        # calculate new sample rate    
        new_sample_rate = Data_dict["sample_rate"]/q
        # creating new time_vec
        new_time_vec = Data_dict["time_vec"][0:-1:q]
        # creating new Data_dict
        new_Data_dict = {"data": new_data, "sample_rate": new_sample_rate, 
                         "n_channels": new_data.shape[1], "ch_labels": Data_dict["ch_labels"], 
                         "time_vec": new_time_vec, "bad_channels": Data_dict["bad_channels"]} 
    return new_Data_dict
    
def merge(Data_dict1,Data_dict2):
    # get data and time_vec from dict1
    data1 = Data_dict1['data']
    time_vec1 = Data_dict1["time_vec"]
    # get data and time_vec from dict1
    data2 = Data_dict2['data']
    time_vec2 = Data_dict2["time_vec"]
    time_vec2 += time_vec1[-1]
    
    # broadcast new_data
    new_data = np.concatenate((data1,data2),axis=0)
    
    # broadcast new_time_vec
    new_time_vec = np.concatenate((time_vec1,time_vec2),axis=0)
    
    # creating new Data_dict
    new_Data_dict = {"data": new_data, "sample_rate": Data_dict1["sample_rate"], 
                     "n_channels": new_data.shape[1], "ch_labels": Data_dict1["ch_labels"], 
                     "time_vec": new_time_vec, "bad_channels": Data_dict1["bad_channels"]}
                     
    return new_Data_dict
    
    
    
    