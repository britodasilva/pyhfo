# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:50:15 2015

@author: anderson
"""

# importing modules
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def plot_eeg(dataset, time_vec, sample_rate, ch_labels = None, start_sec = 0, window_size = 10, amp = 200, figure_size = (15,8), detrend = False, envelope=False, badch = []):
    """
    Function to plot EEG signal 
    Inputs:
        start_sec [0 - default] - the start second to plot (interger)
        window_size [10] - Size of window in second (interger)
        amp [200] - Amplitude between channels in plot (interger)
        figure_size [(15,8)] - Size of figure, tuple of integers with width, height in inches (tuple)
    """
    # Transforming the start_sec in points
    start_sec *= sample_rate
    start_sec = int(start_sec)
    # Transforming the window_size in points
    window_size *= sample_rate
    window_size = int(window_size)
    # Creating the figure 
    f = plt.figure(figsize=figure_size,dpi=600)
    # creating the axes
    sp = f.add_subplot(111)
    # creating a vector with the desired index
    time_window = range(start_sec, start_sec + window_size)
    # declaring tick variables
    ticklocs = []
    ticklabel = [] 
    ch_l = 1
    # Loop to plot each channel
    for ch in range(dataset.shape[1]):
        if ch not in badch:
            # in the axes, plot the raw signal for each channel with a amp diference 
            if detrend:
                sp.plot(time_vec[time_window],(ch_l)*amp + sig.detrend(dataset[time_window,ch]))
            else:
                sp.plot(time_vec[time_window],(ch_l)*amp + dataset[time_window,ch])
            if envelope:
                sp.plot(time_vec[time_window],(ch_l)*amp + np.abs(sig.hilbert(dataset[time_window,ch])))
            # appeng the channel label and the tick location
            if ch_labels is None:
                ticklabel.append(ch_l)            
            else:
                ticklabel.append(ch_labels[ch])
                
            ticklocs.append((ch_l)*amp)
            ch_l += 1

    adjust_spines(sp, ['left', 'bottom'])
    # changing the x-axis (label, limit and ticks)
    plt .xlabel('time (s)', size = 16)
    plt.xlim(time_vec[time_window[0]],time_vec[time_window[-1]])
    plt.xticks(size = 16)
    # changing the y-axis
    plt.yticks(ticklocs, ticklabel, size=16)
    plt.ylim(0,(ch_l)*amp)
    
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 2))  # outward by 10 points
            spine.set_smart_bounds(False)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])  