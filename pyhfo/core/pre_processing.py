# -*- coding: utf-8 -*-
"""
Pre-processing using Data_dict
Created on Fri Apr 17 13:15:57 2015
@author: anderson
"""

import scipy.signal as sig
import numpy as np
from pyhfo.core import DataObj
import matplotlib.pyplot as plt
import itertools

def decimate(Data,q):
    '''
    Use scipy decimate to create a new DataObj with low sample rate data

    Parameters
    ----------
    Data: DataObj
        Data object to resample
    q: int
        Quocient of resample.
    '''
    #reading data
    data = Data.data
    # get shape
    if len(data.shape) == 1:
        nch = 1
        npoints = data.shape[0]
    else:
        npoints, nch = data.shape
    # creating a empty array
    new_data = np.empty((npoints/q,nch))
    new_data[:] = np.NAN
    # decimate each channel
    if nch == 1:
        new_data = sig.decimate(data,q)
    else:
        for ch in range(nch):
            new_data[:,ch] = sig.decimate(data[:,ch],q)
    # calculate new sample rate    
    new_sample_rate = Data.sample_rate/q
    # creating new time_vec
    new_time_vec = Data.time_vec[0:-1:q]
    # creating new Data
    newData = DataObj(new_data,new_sample_rate,Data.amp_unit,
                                Data.ch_labels,new_time_vec,Data.bad_channels)
    return newData
    
def resample(Data,q):
    '''
    Slice data with quocient q. 
    
    Parameters
    ----------
    Data: DataObj
        Data object to resample
    q: int
        Quocient of resample. Sample Rate  should be divisible by q (sample_rat % q = 0)

    '''
    if Data.sample_rate % q != 0:
        print 'sample_rate % q should be int'
        return
    else:
        #reading data
        data = Data.data
        new_data = data[0:-1:q,:]
        # calculate new sample rate    
        new_sample_rate = Data.sample_rate/q
        # creating new time_vec
        new_time_vec = Data.time_vec[0:-1:q]
        # creating new DataObj
        newData = DataObj(new_data,new_sample_rate,Data.amp_unit,Data.ch_labels,new_time_vec,Data.bad_channels)
    return newData

    
def merge(Data1,Data2,new_time = False):
    '''    
    Merging two DataObj
    
    Parameters
    ----------
    Data1: DataObj
        Data object to merge
    Data2: DataObj
        Data object to merge
    new_time: boolean, optional
        False (Defautl) - merge the time_vec from the 2 DataObj
        True - Create a new time_vec starting with 0        
    '''
    
    
    # check if is the same sample rate, then get it
    if Data1.sample_rate != Data2.sample_rate:
        raise Exception('Data object should have same sample_rate')
    sample_rate =  Data1.sample_rate 
    
    # check if is the same number of channels, then get it
    if Data1.n_channels != Data2.n_channels:
        raise Exception('Data object should have same n_channels')
    
    # check if is the same amplitude 
    if Data1.amp_unit != Data2.amp_unit:
        raise Exception('Data object should have same amplitude unit')
    amp_unit = Data1.amp_unit
        
    # get the label from dict 1
    ch_labels = Data1.ch_labels  
    # Append bad channels from both object
    bad_channels = []
    bad_channels =  sorted(set(np.append(bad_channels,Data1.bad_channels)))
    bad_channels =  sorted(set(np.append(bad_channels,Data2.bad_channels)))
     
    
    # get data and time_vec from dict1
    data1 = Data1.data
    time_vec1 = Data1.time_vec
    # get data and time_vec from dict2
    data2 = Data2.data
    time_vec2 = Data2.time_vec
    
    
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
    newData = DataObj(new_data,sample_rate,amp_unit,ch_labels,new_time_vec,bad_channels)
    return newData

def add_bad(Data,channels):
    ''' 
    Add bad channels to the list
    
    Parameters
    ----------
    Data: DataObj
        Data object
    
    channels: str, int or list
        Channels to be added
    '''
    def adding(Data,item):
        if type(item) == str:
            idx = [i for i,x in enumerate(Data.ch_labels) if x == item]
        elif type(item) == int:
            idx = item
        Data.bad_channels = sorted(set(np.append(Data.bad_channels,idx)))
    
    if type(channels) == list:
        for item in channels:
            adding(Data,item)
    else:
        adding(Data,channels)
            
    return Data
    
def remove_bad(Data,channels):
    ''' 
    Remove channels of the bad list
    
    Parameters
    ----------
    Data: DataObj
        Data object
    
    channels: str, int or list
        Channels to be removed
    
    '''
    def removing(Data,item):
        if type(item) == str:
            idx = [i for i,x in enumerate(Data.ch_labels) if x == item]
        elif type(item) == int:
            idx = item
                
        if idx in Data.bad_channels:
            index = [i for i,s in enumerate(Data.bad_channels) if s ==idx]
            Data.bad_channels = np.delete(Data.bad_channels,index)
    
    if type(channels) == list:
        for item in channels:
            removing(Data,item)
    else:
        removing(Data,channels)

    return Data        
    
   
def create_avg(Data):
    ''' 
    Create averege montagem excluding bad_channels
    
    Parameters
    ----------
    Data: DataObj
        Data object
    '''
    # get non-bad channels index
    index = [ch for ch in range(Data.n_channels) if ch not in Data.bad_channels]
    # empty variable    
    avg = np.empty(Data.data.shape)
    avg[:] = np.NAN
    avg_label = []
    for ch in index:
        avg[:,ch] = Data.data[:,ch]-np.mean(Data.data[:,index],1)
    for ch in range(Data.n_channels):
        avg_label.append(Data.ch_labels[ch]+'-avg')
    
    newData = DataObj(avg,Data.sample_rate,Data.amp_unit,avg_label,Data.time_vec,Data.bad_channels)
    return newData
    
    
    
def eegfilt(Data,low_cut = None,high_cut= None,order = None,window = ('kaiser',0.5),whitening=False,filter_test=False, rc = None ,dview = None):
    '''
    Filt EEG Data object with FIR filter.
    
    Parameters
    ----------
    Data: DataObj
        Data object to filt
    low_cut: int
        Low cut frequency. If None, generate a low pass filter with cut 
        frequency in high_cut.
    high_cut: int
        High cut frequency. If None, generate a high pass filter with cut 
        frequency in low_cut.
    order: int, optional
        None (default) - Order of the filter calculated as 1/10 of sample rate
    window : string or tuple of string and parameter values
        Desired window to use. See `scipy.signal.get_window` for a list
        of windows and required parameters.
    '''
    if low_cut == None and high_cut == None:
        raise Exception('You should determine the cutting frequencies')
    if whitening:
        signal = np.diff(Data.data,axis=0)
        time_vec = Data.time_vec[:-1]
    else:
        signal = Data.data
        time_vec = Data.time_vec
    sample_rate = Data.sample_rate
    labels = Data.ch_labels
    if len(signal.shape) == 1:
        nch = 1
        npoints = signal.shape[0]
       
    else:
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
    
    if filter_test:
        w,h = sig.freqz(b)
        fig = plt.figure(figsize=(10,10))
        plt.title('Digital filter frequency response')
        ax1 = fig.add_subplot(111)
        plt.plot(w, 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        plt.plot(w, angles, 'g')
        plt.ylabel('Angle (radians)', color='g')
        plt.grid()
        plt.axis('tight')   
        plt.show()
       
        return
    # Creating filtered, numpy array with the filtered signal of raw data
    filtered = np.empty((npoints,nch))
    filtered[:] = np.NAN
    if nch == 1:
        print 'Filtering channel'
        filtered = sig.filtfilt(b,np.array([1]),sig.detrend(signal))
    else:
        if rc is not None:
            nch -= len(Data.bad_channels) 
            sig_det = sig.detrend(signal[:,[ch for ch in range(nch) if ch not in Data.bad_channels]])
            fir = dview.map_sync(sig.filtfilt,itertools.repeat(b,nch),itertools.repeat(np.array([1]),nch),sig_det.T)
            filtered = np.asarray(fir).T
        else:
            for ch in range(nch):
                if ch not in Data.bad_channels:
                    print 'Filtering channel ' + labels[ch]
                    filtered[:,ch] = sig.filtfilt(b,np.array([1]),sig.detrend(signal[:,ch]))
      
    newData = DataObj(filtered,sample_rate,Data.amp_unit,labels,time_vec,Data.bad_channels)
    return newData        
    
def pop_channel(Data,ch):
    '''
    Create a new DataObj just with the the choose channel
    
    Parameters
    ----------
    Data: DataObj
        Data object to filt
    ch: int
        Channel number
    '''
    if len(Data.data.shape) == 1:
         raise Exception('DataObj is single channel')
    signal = Data.data[:,ch]
    label = Data.ch_labels[ch]
    newData = DataObj(signal,Data.sample_rate,Data.amp_unit,label,Data.time_vec)
    return newData