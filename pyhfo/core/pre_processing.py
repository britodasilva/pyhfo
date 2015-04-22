# -*- coding: utf-8 -*-
"""
Pre-processing using Data_dict
Created on Fri Apr 17 13:15:57 2015
@author: anderson
"""

import scipy.signal as sig
import numpy as np
from pyhfo.io import make_dict   

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
    new_Data_dict = make_dict(new_data,new_sample_rate,nch,Data_dict["ch_labels"],new_time_vec,Data_dict["bad_channels"])
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
        
        new_Data_dict = make_dict(new_data,new_sample_rate,new_data.shape[1],Data_dict["ch_labels"],new_time_vec,Data_dict["bad_channels"])
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
    new_Data_dict = make_dict(new_data,Data_dict1["sample_rate"],new_data.shape[1],Data_dict1["ch_labels"],new_time_vec,Data_dict1["bad_channels"])
    return new_Data_dict

def add_bad(Data_dict,channels):
    ''' 
    Add bad channels to the list
    '''
    def adding(Data_dict,item):
        if type(item) == str:
            idx = [i for i,x in enumerate(Data_dict['ch_labels']) if x == item]
        elif type(item) == int:
            idx = item
        Data_dict['bad_channels'] = sorted(set(np.append(Data_dict['bad_channels'],idx)))
    
    if type(channels) == list:
        for item in channels:
            adding(Data_dict,item)
    else:
        adding(Data_dict,channels)
            
    return Data_dict
    
def remove_bad(Data_dict,channels):
    ''' 
    Remove channels of the bad list
    '''
    def removing(Data_dict,item):
        if type(item) == str:
            idx = [i for i,x in enumerate(Data_dict['ch_labels']) if x == item]
        elif type(item) == int:
            idx = item
                
        if idx in Data_dict['bad_channels']:
            index = [i for i,s in enumerate(Data_dict['bad_channels']) if s ==idx]
            Data_dict['bad_channels'] = np.delete(Data_dict['bad_channels'],index)
    
    if type(channels) == list:
        for item in channels:
            removing(Data_dict,item)
    else:
        removing(Data_dict,channels)

    return Data_dict        
    
   
def create_avg(Data_dict):
    ''' 
    Get Data_dict, make averege montagem excluding bad_channels
    '''
    # get non-bad channels index
    index = [ch for ch in range(Data_dict['n_channels']) if ch not in Data_dict['bad_channels']]
    # empty variable    
    avg = np.empty(Data_dict['data'].shape)
    avg[:] = np.NAN
    avg_label = []
    for ch in index:
        avg[:,ch] = Data_dict['data'][:,ch]-np.mean(Data_dict['data'][:,index],1)
    for ch in range(Data_dict['n_channels']):
        avg_label.append(Data_dict["ch_labels"][ch]+'-avg')
    
    new_Data_dict = make_dict(avg,Data_dict["sample_rate"],avg.shape[1],avg_label,Data_dict['time_vec'],Data_dict["bad_channels"])
    return new_Data_dict
    
    
    