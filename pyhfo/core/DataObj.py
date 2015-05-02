# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:46:39 2015

@author: anderson
"""
from pyhfo.ui import plot_eeg
import numpy as np

class DataObj(object):
    ''' 
    Data object
    
    Parameters
    ----------
    data : numpy array 
        Array with shape=(points,channels)
    sample_rate: int
        Sample rate in Hz.
    amp_unit: str
        Sting with amplitude string
    ch_labels: list of strings, optional
        None (default) - labels will be numbers starting from 0
        ist of strings - List with channels labels
    time_vec: numpy array, optional
        None (default) - create a time vector starting from zero
        Array shape=(points,) with time vector                    
    bad_channels: list of int, optional
        None (default) - No bad channels
        List - create a list of bad channels. 
    '''
    htype = 'Data'
    def __init__(self,data,sample_rate,amp_unit,ch_labels=None,time_vec=None,bad_channels=None):
        self.npoints, self.n_channels = data.shape
        if self.npoints < self.n_channels: 
            raise Exception('data should be numpy array points x channels')        
        self.data = data
        self.sample_rate = sample_rate
        self.amp_unit = amp_unit
        if ch_labels == None:
            self.ch_labels = range(self.n_channels)
        else:
            self.ch_labels = ch_labels
        if time_vec==None:
            end_time  = self.n_points/self.sample_rate
            self.time_vec  = np.linspace(0,end_time,self.n_points,endpoint=False)
        else:
            if time_vec.shape[0] != self.npoints:
                raise Exception('time_vec and data should have same number of points')
            self.time_vec = time_vec
        if bad_channels == None:
            self.bad_channels = []
        else:
            self.bad_channels = bad_channels
    
    def plot(self,*param,**kwargs):
        plot_eeg(self,*param,**kwargs)