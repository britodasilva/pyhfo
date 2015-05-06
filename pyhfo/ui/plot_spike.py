# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:11:16 2015

@author: anderson
"""

# importing modules
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from . import adjust_spines

def plot_single_spk(spk,subplot = None, spines = ['left', 'bottom'],
                    figure_size = (5,5),dpi=600,**kwargs):
    """
    Function to plot single Spike
    
    Parameters
    ----------
    spk: SpikeObj
        Spike object to plot
    subplot: matplotlib axes 
        None (default) - create a new figure
        ax - axes of figure where figure should plot
    spines: str
        ['left', 'bottom'] (default) - plot figure with left and bottom spines only
    figure_size: tuple
        (5,5) (default) - Size of figure, tuple of integers with width, height in inches 
    dpi: int
        600 - DPI resolution
    **kwargs: matplotlib arguments
    """
    if subplot == None:    
        # Creating the figure 
        f = plt.figure(figsize=figure_size,dpi=dpi)
        # creating the axes
        ax = f.add_subplot(111)
    else:
        ax = subplot
    ax.plot(range(-20,44),spk.waveform,**kwargs)
    adjust_spines(ax, spines)
    
    
    
    
def plot_spk_cluster(evlist,cluster,color='b',ax = None, spines = [], plot_mean = True,figure_size=(5,5),dpi=600):
    """
    Function to plot cluster of spikes
    
    Parameters
    ----------
    evlist: EventList
        EventList object to plot
    cluster: int
        Number of the cluster
    color: str
        Color of plot
    spines: str
        ['left', 'bottom'] (default) - plot figure with left and bottom spines only
    plot_mean: boolean
        True (default) - plot mean line
    figure_size: tuple
        (5,5) (default) - Size of figure, tuple of integers with width, height in inches 
    dpi: int
        600 - DPI resolution
    """
    if ax == None:
        # Creating the figure 
        f = plt.figure(figsize=figure_size,dpi=dpi)
        # creating the axes
        ax = f.add_subplot(111)
    spikes = np.array([]) # creating a empty array 
     
    objs = [x for x in evlist.event if x.cluster == cluster]
    npspk, = objs[0].waveform.shape
    for sp in objs:
        ax.plot(range(-20,44),sp.waveform,color=color,lw=0.5)
        spikes = np.append(spikes, sp.waveform)
        
    if plot_mean and len(evlist.event)>1:
        spikes = spikes.reshape(len(objs),npspk)
        ax.plot(range(-20,44),np.mean(spikes,axis=0),'k',lw=2)
        ax.plot(range(-20,44),np.mean(spikes,axis=0)-np.std(spikes,axis=0),'k',lw=1)
        ax.plot(range(-20,44),np.mean(spikes,axis=0)+np.std(spikes,axis=0),'k',lw=1)
    adjust_spines(ax, spines)

    

    



    