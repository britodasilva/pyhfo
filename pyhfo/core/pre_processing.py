# -*- coding: utf-8 -*-
"""
Pre-processing using Data_dict
Created on Fri Apr 17 13:15:57 2015
@author: anderson
"""

import scipy.signal as sig
import numpy as np
from pyhfo.core import create_DataObj


def decimate(DataObj,q):
    '''
    Use scipy decimate to create a new DataObj with low sample rate data
    '''
    #reading data
    data = DataObj.data
    # get shape
    npoints, nch = DataObj.data.shape
    # creating a empty array
    new_data = np.empty((npoints/q,nch))
    new_data[:] = np.NAN
    # decimate each channel
    for ch in range(nch):
        new_data[:,ch] = sig.decimate(data[:,ch],q)
    # calculate new sample rate    
    new_sample_rate = DataObj.sample_rate/q
    # creating new time_vec
    new_time_vec = DataObj.time_vec[0:-1:q]
    # creating new DataObj
    newDataObj = create_DataObj(new_data,new_sample_rate,DataObj.amp_unit,
                                DataObj.ch_labels,new_time_vec,DataObj.bad_channels)
    return newDataObj
    
def resample(DataObj,q):
    '''
    Slice data with quocient q. sample_rate % q should be 0
    '''
    if DataObj.sample_rate % q != 0:
        print 'sample_rate % q should be int'
        return
    else:
        #reading data
        data = DataObj.data
        new_data = data[0:-1:q,:]
        # calculate new sample rate    
        new_sample_rate = DataObj.sample_rate/q
        # creating new time_vec
        new_time_vec = DataObj.time_vec[0:-1:q]
        # creating new DataObj
        newDataObj = create_DataObj(new_data,new_sample_rate,DataObj.amp_unit,DataObj.ch_labels,new_time_vec,DataObj.bad_channels)
    return newDataObj

    
def merge(DataObj1,DataObj2,new_time = False):
    # check if is the same sample rate, then get it
    if DataObj1.sample_rate != DataObj2.sample_rate:
        raise Exception('Data object should have same sample_rate')
    sample_rate =  DataObj1.sample_rate 
    
    # check if is the same number of channels, then get it
    if DataObj1.n_channels != DataObj2.n_channels:
        raise Exception('Data object should have same n_channels')
    
    # check if is the same amplitude 
    if DataObj1.amp_unit != DataObj2.amp_unit:
        raise Exception('Data object should have same amplitude unit')
    amp_unit = DataObj1.amp_unit
        
    # get the label from dict 1
    ch_labels = DataObj1.ch_labels  
    # Append bad channels from both object
    bad_channels = []
    bad_channels =  sorted(set(np.append(bad_channels,DataObj1.bad_channels)))
    bad_channels =  sorted(set(np.append(bad_channels,DataObj2.bad_channels)))
     
    
    # get data and time_vec from dict1
    data1 = DataObj1.data
    time_vec1 = DataObj1.time_vec
    # get data and time_vec from dict2
    data2 = DataObj2.data
    time_vec2 = DataObj2.time_vec
    
    
    # broadcast new_data
    new_data = np.concatenate((data1,data2),axis=0)
    
    
    if new_time:
        n_points         = new_data.shape[0]
        end_time         = n_points/sample_rate
        new_time_vec = np.linspace(0,end_time,n_points,endpoint=False)
    else:
        # broadcast new_time_vec
        new_time_vec = np.concatenate((time_vec1,time_vec2),axis=0)
    
    # creating new DataObj
    newDataObj = create_DataObj(new_data,sample_rate,amp_unit,ch_labels,new_time_vec,bad_channels)
    return newDataObj

def add_bad(DataObj,channels):
    ''' 
    Add bad channels to the list
    '''
    def adding(DataObj,item):
        if type(item) == str:
            idx = [i for i,x in enumerate(DataObj.ch_labels) if x == item]
        elif type(item) == int:
            idx = item
        DataObj.bad_channels = sorted(set(np.append(DataObj.bad_channels,idx)))
    
    if type(channels) == list:
        for item in channels:
            adding(DataObj,item)
    else:
        adding(DataObj,channels)
            
    return DataObj
    
def remove_bad(DataObj,channels):
    ''' 
    Remove channels of the bad list
    '''
    def removing(DataObj,item):
        if type(item) == str:
            idx = [i for i,x in enumerate(DataObj.ch_labels) if x == item]
        elif type(item) == int:
            idx = item
                
        if idx in DataObj.bad_channels:
            index = [i for i,s in enumerate(DataObj.bad_channels) if s ==idx]
            DataObj.bad_channels = np.delete(DataObj.bad_channels,index)
    
    if type(channels) == list:
        for item in channels:
            removing(DataObj,item)
    else:
        removing(DataObj,channels)

    return DataObj        
    
   
def create_avg(DataObj):
    ''' 
    Get Data_dict, make averege montagem excluding bad_channels
    '''
    # get non-bad channels index
    index = [ch for ch in range(DataObj.n_channels) if ch not in DataObj.bad_channels]
    # empty variable    
    avg = np.empty(DataObj.data.shape)
    avg[:] = np.NAN
    avg_label = []
    for ch in index:
        avg[:,ch] = DataObj.data[:,ch]-np.mean(DataObj.data[:,index],1)
    for ch in range(DataObj.n_channels):
        avg_label.append(DataObj.ch_labels[ch]+'-avg')
    
    newDataObj = create_DataObj(avg,DataObj.sample_rate,DataObj.amp_unit,avg_label,DataObj.time_vec,DataObj.bad_channels)
    return newDataObj
    
    
    
def eegfilt(DataObj,low_cut = None,high_cut= None,order = None,window = ('kaiser',0.5)):
    '''
    Filter EEG Data dict. Create a high pass filter if only have low_cut, a low pass filter if only has a high_cut and a pass band filter if has both. 
    '''
    if low_cut == None and high_cut == None:
        raise Exception('You should determine the cutting frequencies')
       
    signal = DataObj.data
    sample_rate = DataObj.sample_rate
    time_vec = DataObj.time_vec
    labels = DataObj.ch_labels
    npoints, nch = signal.shape
    # order
    if order == None:
        numtaps = int(sample_rate/10 + 1)
    else:
        numtaps = order
    # Nyquist rate
    nyq = sample_rate/2
    
    # cutoff frequencies
    if high_cut == None: # high pass
        f = [low_cut]
        pass_zero=False
    elif low_cut == None: # low pass
        f = [high_cut]
        pass_zero=True        
    else: # band pass
        f = [low_cut,high_cut]
        pass_zero=False
    
    # Creating filter
    b = sig.firwin(numtaps,f,pass_zero=pass_zero,window=window,nyq=nyq)
    # Creating filtered, numpy array with the filtered signal of raw data
    filtered = np.empty((npoints,nch))
    filtered[:] = np.NAN
    for ch in range(nch):
        if ch not in DataObj.bad_channels:
            print 'Filtering channel ' + labels[ch]
            filtered[:,ch] = sig.filtfilt(b,np.array([1]),signal[:,ch])
            
    newDataObj = create_DataObj(filtered,sample_rate,DataObj.amp_unit,labels,time_vec,DataObj.bad_channel)
    return newDataObj        
    
    