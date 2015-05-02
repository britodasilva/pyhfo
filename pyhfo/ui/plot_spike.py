# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:11:16 2015

@author: anderson
"""

# importing modules
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    ax.plot(spk.waveform,**kwargs)
    adjust_spines(ax, spines)
    
    
    
    
def plot_cluster(self,cluster,color='b',ax = None, spines = [], plot_mean = True,figure_size=(5,5),dpi=600):
    if len(self.event) == 0:
        raise Exception('No events to plot')
    if ax == None:
        # Creating the figure 
        f = plt.figure(figsize=figure_size,dpi=dpi)
        # creating the axes
        ax = f.add_subplot(111)
    spikes = np.array([]) # creating a empty array 
     
    objs = [x for x in self.event if x.cluster == cluster]
    npspk, = objs[0].waveform.shape
    for sp in objs:
        ax.plot(range(npspk),sp.waveform,color=color,lw=0.5)
        spikes = np.append(spikes, sp.waveform)
        
    if plot_mean and len(self.event)>1:
        spikes = spikes.reshape(len(objs),npspk)
        ax.plot(range(npspk),np.mean(spikes,axis=0),'k',lw=2)
        ax.plot(range(npspk),np.mean(spikes,axis=0)-np.std(spikes,axis=0),'k',lw=1)
        ax.plot(range(npspk),np.mean(spikes,axis=0)+np.std(spikes,axis=0),'k',lw=1)
    adjust_spines(ax, spines)

    
def plot_all_clusters(self,plot_mean = True,figure_size=(10,10),dpi=600):
    if len(self.event) == 0:
        raise Exception('No events to plot')
    cluster = self.__getcluster__()
    num_clus = int(np.max(cluster))+1
    ncols = int(math.ceil(math.sqrt(num_clus)))
    nrows = int(math.floor(math.sqrt(num_clus)))
    fig,sb = plt.subplots(nrows,ncols,sharey=True,figsize=figure_size,dpi=dpi)
    c = 0
    l = 0
    for clus in range(num_clus):
        if c == ncols:
            c = 0
            l += 1
        self.plot_cluster(clus, ax = sb[l,c])
        sb[l,c].set_title('Cluster ' + str(clus))
        c +=1
    


def rastergram(self, ax = None, spines = ['left'],time_vec = None,figure_size=(15,5),dpi=600):
    if ax == None:
         # Creating the figure 
        f = plt.figure(figsize=figure_size,dpi=dpi)
        # creating the axes
        ax = f.add_subplot(111)
    cluster = self.__getcluster__()
    num_clus = int(np.max(cluster))+1
    label = []
    for clus in range(num_clus):
        label.append('Cluster ' + str(clus))
        objs = [x for x in self.event if x.cluster == clus]
        for ev in objs:
            rect = patches.Rectangle((ev.tstamp,clus),0.001,.8, lw=0.5) 
            ax.add_patch(rect)
    tstamp = self.__gettstamp__()
    ax.set_ylim(0,num_clus)
    if time_vec != None:
        ax.set_xlim(time_vec[0],time_vec[-1])
    else:
        ax.set_xlim(tstamp[0],tstamp[-1])
    plt.yticks(np.arange(num_clus)+0.5,label, size=16)
    adjust_spines(ax, spines)
    