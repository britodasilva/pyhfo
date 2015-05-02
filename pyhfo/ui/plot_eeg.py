# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:16:21 2015

@author: anderson
"""

# importing modules
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from . import adjust_spines

def plot_eeg(Data,start_sec = 0, window_size = 10, amp = 200, figure_size = (15,8),
             dpi=600, detrend = True, envelope=False, plot_bad = False, exclude = [], grid=True, 
             xtickspace = 1,saveplot = None, subplot = None ,spines = ['left', 'bottom'],**kwargs):
    """
    Function to plot EEG signal 
    Inputs:
        Data_dict - data dictionary  
        time_vec - numpy array with time vector
        sample_rate : sample rate in Hz
        start_sec [0 - default] - the start second to plot (interger)
        window_size [10] - Size of window in second (interger)
        amp [200] - Amplitude between channels in plot (interger)
        figure_size [(15,8)] - Size of figure, tuple of integers with width, height in inches (tuple)
        detrend [False] - detrend each line before filter
        envelop [False] - plot the amplitude envelope by hilbert transform
        plot_bad [False] - if False, exclude bad channels from plot
        exclude - list of channels to exclude
        gride [True] - plot grid
        xtickspace [1] - distance of tick
        saveplot [None] - String with the name to save (ex: 'Figura.png')
        subplot [None] - axes of figure where figure should plot
        spines ['left', 'bottom'] 
    """
    #geting data from Data_dict
    data = Data.data
    time_vec    = Data.time_vec
    sample_rate = Data.sample_rate
    ch_labels   = Data.ch_labels
    if plot_bad:
        badch = np.array([],dtype=int) # a empty array 
    else:
        badch = Data.bad_channels
     
    if type(exclude) == list:
       for item in exclude:
            if type(item) == str:
                idx = [i for i,x in enumerate(ch_labels) if x == item]
                badch = sorted(set(np.append(badch,idx)))
            elif type(item) == int:
                idx = item
                badch = sorted(set(np.append(badch,idx)))

    elif type(exclude) == str:
        idx = [i for i,x in enumerate(ch_labels) if x == exclude]
        badch = sorted(set(np.append(badch,idx)))
    elif type(exclude) == int:
        idx = exclude
        badch = sorted(set(np.append(badch,idx)))
    
    # Transforming the start_sec in points
    start_sec *= sample_rate
    start_sec = int(start_sec)
    # Transforming the window_size in points
    window_size *= sample_rate
    window_size = int(window_size)
    if subplot == None:    
        # Creating the figure 
        f = plt.figure(figsize=figure_size,dpi=dpi)
        # creating the axes
        sp = f.add_subplot(111)
    else:
        sp = subplot
    # creating a vector with the desired index
    time_window = range(start_sec, start_sec + window_size)
    # declaring tick variables
    yticklocs = []
    yticklabel = [] 
    ch_l = 1
    # Loop to plot each channel
    for ch in [x for x in range(data.shape[1]) if x not in badch]:
        # in the axes, plot the raw signal for each channel with a amp diference 
        if detrend:
            sp.plot(time_vec[time_window],(ch_l)*amp + sig.detrend(data[time_window,ch]),**kwargs)
        else:
            sp.plot(time_vec[time_window],(ch_l)*amp + data[time_window,ch],**kwargs)
        if envelope:
            sp.plot(time_vec[time_window],(ch_l)*amp + np.abs(sig.hilbert(data[time_window,ch])),**kwargs)
        # appeng the channel label and the tick location
        if ch_labels is None:
            yticklabel.append(ch_l)            
        else:
            yticklabel.append(ch_labels[ch])
            
        yticklocs.append((ch_l)*amp)
        ch_l += 1

    adjust_spines(sp, spines)
    if len(spines) > 0:
        # changing the x-axis (label, limit and ticks)
        plt .xlabel('time (s)', size = 16)
        #xtickslocs = np.linspace(int(time_vec[time_window[0]]),int(time_vec[time_window[-1]]),int(window_size/(sample_rate*xtickspace)),endpoint=True)
        xtickslocs = np.arange(int(time_vec[time_window[0]]),int(time_vec[time_window[-1]]*2),xtickspace)     
        xtickslabels = ['']*len(xtickslocs)
        for i,x in enumerate(np.arange(0,len(xtickslocs),10)):
            xtickslabels[i*10] = xtickslocs[x]
        plt.xticks(xtickslocs,xtickslabels,size = 16)
        # changing the y-axis
        plt.yticks(yticklocs, yticklabel, size=16)
    
    if grid:    
        ax = plt.gca()
        ax.xaxis.grid(True)
    
    sp.set_xlim(time_vec[time_window[0]],time_vec[time_window[-1]]+np.diff(time_vec[time_window[0:2]]))
    #sp.set_ylim(0,(ch_l)*amp)
    
    if saveplot != None:
        if type(saveplot) == str: 
            plt.savefig(saveplot, bbox_inches='tight')
        else:
            raise Exception('saveplot should be a string')

