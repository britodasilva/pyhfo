# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:12:56 2015

@author: anderson
"""

# importing modules
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from . import adjust_spines

def plot_single_hfo(hfo, envelope = False, xlim =[-1,1], cutoff = None, v = True,
                    axes = None, figure_size = (15,10),dpi=600,saveplot = None):
    """
    Function to plot single Spike
    
    Parameters
    ----------
    hfo: HFOObj
        HFO object to plot
    envelope: boolean
        True (default) - plot envelope of filtered signal
    figure_size: tuple
        (5,5) (default) - Size of figure, tuple of integers with width, height in inches 
    dpi: int
        600 - DPI resolution
    """
    if axes == None:
        # Creating the figure 
        fig = plt.figure(figsize=figure_size,dpi=dpi)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
    else:
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

    # number of points
    npoints = hfo.waveform.shape[0]
    time_v = np.linspace(-1,1,npoints,endpoint=True)
    # creating the axes
    
    ax1.plot(time_v,hfo.waveform[:,0],'b')
    ax1.plot(time_v[hfo.start_idx:hfo.end_idx],hfo.waveform[hfo.start_idx:hfo.end_idx,0],'k')
    
    adjust_spines(ax1, ['left'])
    ax1.set_xlim(xlim)
    
    
    
    filt = hfo.waveform[:,1]
    ax2.plot(time_v,filt)    
    ax2.plot(time_v[hfo.start_idx:hfo.end_idx],filt[hfo.start_idx:hfo.end_idx],'k')
    if envelope:
        env = hfo.waveform[:,2]
        ax4 = ax2.twinx()
        ax4.plot(time_v,env,'g')
    

    
    adjust_spines(ax2, ['left', 'bottom'])
    ax2.set_xlim(xlim)
    
    
    hfo.spectrum.plot(cutoff = cutoff, v = v, ax = ax3)
    ax3.set_title('peak freq = ' + str(hfo.spectrum.peak_freq))
    adjust_spines(ax3, ['left', 'bottom'])
    
    if saveplot != None:
        if type(saveplot) == str: 
            plt.savefig(saveplot, bbox_inches='tight')
        else:
            raise Exception('saveplot should be a string')
    plt.draw()     
            
def plot_mean_hfo(evlist,color='blue',  xlim =[-1,1], figure_size=(10,10),dpi=600,saveplot = None):
    """
    Function to plot cluster of HFOs
    
    Parameters
    ----------
    evlist: EventList
        EventList object to plot
    color: str
        Color of plot

    figure_size: tuple
        (5,5) (default) - Size of figure, tuple of integers with width, height in inches 
    dpi: int
        600 - DPI resolution
    """
    f = plt.figure(figsize=figure_size,dpi=dpi)
    
    
    
    
    raw = np.array([]) # creating a empty array 
    filt = np.array([]) # creating a empty array
    pxx = np.array([]) # creating a empty array
    nwave, a = evlist[0].waveform.shape
    time_v = np.linspace(-1,1,nwave,endpoint=True)
    npw, = evlist[0].spectrum.nPxx.shape
    F = evlist[0].spectrum.F
    for hfo in evlist:
        raw = np.append(raw, hfo.waveform[:,0])
        #ax1.plot(time_v,hfo.waveform[:,0],lw=.5)
        filt = np.append(filt, hfo.waveform[:,1])
        #ax2.plot(time_v,hfo.waveform[:,1],lw=.5)
        pxx = np.append(pxx, hfo.spectrum.nPxx)
        
    raw = raw.reshape(len(evlist),nwave)
    filt = filt.reshape(len(evlist),nwave)
    pxx = pxx.reshape(len(evlist),npw)

    
    
   
    ax1 = plt.subplot(311)
    m = np.mean(raw,0)
    s = np.std(raw,0)/np.sqrt(raw.shape[0])
    plt.plot(time_v,m,'k',lw=2)
    #ax1.fill_between(time_v,m+s,m-s, facecolor=color, alpha=0.1)
    ax1.set_xlim(xlim)
    adjust_spines(ax1, ['left'])
    
    ax2 = plt.subplot(312)
    m = np.mean(filt,0)
    s = np.std(filt,0)/np.sqrt(filt.shape[0])
    plt.plot(time_v,m,'k',lw=2)
    #ax2.fill_between(time_v,m+s,m-s, facecolor=color, alpha=0.1)
    ax2.set_xlim(xlim)
    adjust_spines(ax2, ['left', 'bottom'])
    
    ax3 = plt.subplot(313)
    m = np.mean(pxx,0)
    s = np.std(pxx,0)/np.sqrt(pxx.shape[0])
    plt.plot(F,m,'k',lw=2)
    ax3.fill_between(F,m+s,m-s, facecolor=color, alpha=0.1)
    adjust_spines(ax3, ['left', 'bottom'])
    
    if saveplot != None:
        if type(saveplot) == str: 
            plt.savefig(saveplot, bbox_inches='tight')
        else:
            raise Exception('saveplot should be a string')