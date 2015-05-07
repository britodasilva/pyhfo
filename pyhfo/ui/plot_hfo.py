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

def plot_single_hfo(hfo, envelope = True, 
                    figure_size = (15,10),dpi=600):
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
   
    # Creating the figure 
    f = plt.figure(figsize=figure_size,dpi=dpi)
    # number of points
    npoints = hfo.waveform.shape[0]
    # creating the axes
    ax1 = f.add_subplot(311)
    ax1.plot(range(npoints),hfo.waveform[:,0],'b')
    ax1.plot(range(hfo.start_idx,hfo.end_idx),hfo.waveform[hfo.start_idx:hfo.end_idx,0],'k')
    adjust_spines(ax1, ['left'])
    
    
    
    ax2 = f.add_subplot(312)
    filt = hfo.waveform[:,1]
    ax2.plot(range(npoints),filt)    
    ax2.plot(range(hfo.start_idx,hfo.end_idx),filt[hfo.start_idx:hfo.end_idx],'k')
    if envelope:
        ax2.plot(range(npoints),np.abs(sig.hilbert(filt)))
    


    adjust_spines(ax2, ['left', 'bottom'])
    
    ax3 = f.add_subplot(313)
    
    signal = sig.detrend(hfo.waveform[hfo.start_idx:hfo.end_idx,0]) # detrending
    next2power = 2**(hfo.sample_rate-1).bit_length() # next power of two of sample rate (power of 2 which contains at least 1 seg)
    signal = np.lib.pad(signal, int((next2power-len(signal))/2), 'constant', constant_values=0)
    F, Pxx = sig.welch(np.diff(signal), fs = hfo.sample_rate, nperseg = next2power)
    Pxx = Pxx/np.sum(Pxx)
    plt.plot(F,Pxx)
    peakFreaq = F[np.argmax(Pxx)]
    ax3.set_title('peak freq = ' + str(peakFreaq))
    adjust_spines(ax3, ['left', 'bottom'])
    
    
