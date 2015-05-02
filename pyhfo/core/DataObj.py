# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:46:39 2015

@author: anderson
"""
from pyhfo.ui import plot_eeg
import numpy as np
import matplotlib.pyplot as plt
from IPython.html import widgets # Widget definitions
from IPython.display import display, clear_output # Used to display widgets in the notebook


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




    

    
    def plot(self,start_sec = 0, window_size = 10, figure_size = (15,8),
             dpi=600,**kwargs):
        
                 
        def f_button(clicked):
            start = test.add(10)
            clear_output()
            plot_eeg(self,start,window_size,**kwargs)
        
        def b_button(clicked):
            start = test.add(-10)
            clear_output()
            plot_eeg(self,start,window_size,**kwargs)        
        
        # Creating the figure 
        f = plt.figure(figsize=figure_size,dpi=dpi)
        # creating the axes
        ax = f.add_subplot(111)
        if 'start_sec' in kwargs:
            start_sec = kwargs['start_sec']
        else:
            start_sec = 0
        plot_eeg(self,start_sec,window_size,subplot = ax, **kwargs)
        #plt.close(fig)
        test = Index(start_sec)
            
        buttonf = widgets.Button(description = ">>")
        buttonb = widgets.Button(description = "<<")
            
        buttonf.on_click(f_button)
        buttonb.on_click(b_button)
        vbox = widgets.Box()
        vbox.children = [buttonb,buttonf]        
        display(vbox)

            
        
class Index(object):
    def __init__(self,start_sec):
        self.ind = start_sec
    def add(self,num):
        self.ind += num
        return self.ind
            
            
        
        
