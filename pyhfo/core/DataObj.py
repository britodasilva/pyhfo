# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:46:39 2015

@author: anderson
"""
from pyhfo.ui import plot_eeg
from pyhfo.io.o_functions import save_dataset
from .IndexObj import IndexObj
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    def __init__(self,data,sample_rate,amp_unit,ch_labels=None,time_vec=None,bad_channels=None,file_name=None,dataset_name = None):
        if len(data.shape) == 1:
            self.n_channels = 1
            self.npoints = data.shape[0]
        else:
            self.npoints, self.n_channels = data.shape
        if self.npoints < self.n_channels: 
            raise Exception('data should be numpy array points x channels')        
        self.data = data
        self.sample_rate = int(sample_rate)
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
        if file_name != None:
            self.filename = file_name
        if dataset_name != None:
            self.datasetname = dataset_name
   
    def plot(self,start_sec = 0, window_size = 10, amp = 100, figure_size = (15,8),
             dpi=600,events = None, **kwargs):
        ''' 
        plot DataObj
        
        Parameters
        ----------
        start_sec: int, optional
            0  (defaut) - The start second from begining of file to plot (n+time_vev[1] to start at second n) 
        window_size: int, optional
            10 (default) - Size of window in second 
        amp: int
            100 (default) - Amplitude between channels in plot 
        figure_size: tuple
            (15,8) (default) - Size of figure, tuple of integers with width, height in inches 
        dpi: int
            600 - DPI resolution
        detrend: boolean  
            False (default) - detrend each line before filter
        envelop: boolean 
            False (default) - plot the amplitude envelope by hilbert transform
        plot_bad: boolean
            False (default) - exclude bad channels from plot
        exclude: list 
            Channels to exclude from plot
        gride: boolean
            True (default) - plot grid
        xtickspace: int 
            1 (default) - distance of tick in seconds
        saveplot: str
            None (default) - Don't save
            String with the name to save (ex: 'Figura.png')
        subplot: matplotlib axes 
            None (default) - create a new figure
            ax - axes of figure where figure should plot
        spines: str
            ['left', 'bottom'] (default) - plot figure with left and bottom spines only
        **kwargs: matplotlib arguments        
        '''
        def plot_events(self,start,window_size,amp,s,e,ax):
            color = ['blue','red']
            for co, ev in enumerate(events):
                tsp = ev.__getlist__('tstamp')
                for idx, x in enumerate(tsp):
                    if x > s and x < e:
                        #print (ev.event[idx].start_sec,ev.event[idx].channel+.5),ev.event[idx].duration,amp
                        rect = patches.Rectangle((ev.event[idx].start_sec,(ev.event[idx].channel+.5)*amp),ev.event[idx].duration,amp, lw=2,alpha=0.1,facecolor=color[co]) 
                        ax.add_patch(rect)
                        
                    
            
        def ploting(self,start, ap):
            clear_output()
            start_sec, end_sec, ax = plot_eeg(self,start,window_size,amp = ap, figure_size=figure_size,dpi=dpi,time_out = True, **kwargs)
            if events != None:
                plot_events(self,start,window_size,amp = ap,s=start_sec, e=end_sec, ax=ax)
        def f_button(clicked):
            start = sec.add(window_size)
            aplit = apl.ind
            ploting(self,start,aplit)
            
        def b_button(clicked):
            start = sec.add(-window_size)
            aplit = apl.ind
            ploting(self,start,aplit)
            
        def u_button(clicked):
            start = sec.ind
            aplit = apl.mul(2)
            ploting(self,start,aplit)
        
        def d_button(clicked):
            start = sec.ind
            aplit = apl.div(2)
            ploting(self,start,aplit)
        

        ploting(self,start_sec,amp)
        #plt.close(fig)
        sec = IndexObj(start_sec)
        apl = IndexObj(amp)
            
        buttonf = widgets.Button(description = ">>")
        buttonb = widgets.Button(description = "<<")
        buttonup = widgets.Button(description = "^")
        buttondown = widgets.Button(description = "v")
        
        buttonf.on_click(f_button)
        buttonb.on_click(b_button)
        buttonup.on_click(u_button)
        buttondown.on_click(d_button)
        
        hbox = widgets.HBox()
        hbox.children = [buttonup,buttondown] 
        vbox = widgets.Box()
        vbox.children = [buttonb,buttonf,hbox]        
        display(vbox)

    
    def save(self,filename = None,datasetname = None ):
        if filename == None:
            if hasattr(self,'filename'):
                filename = self.filename
                
            else:
                raise Exception('No file name')
                
        if datasetname == None:
            if hasattr(self,'datasetname'):
                datasetname = self.datasetname
            else:
                raise Exception('No dataset name')        
     
        save_dataset(self,filename,datasetname)            


            
            
        
        
