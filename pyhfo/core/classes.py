# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:00:13 2015

@author: anderson
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyhfo.ui import adjust_spines
from pyhfo.ui import plot_eeg
import numpy as np
import math


class DataObj(object):
    ''' 
    Create a Data object
    data: numpy array - points x channels
    sample_rate: int
    amp_unit: str
    n_channes: int
    ch_labels: list of strings
    time_vec: numpy array - (points,)
    bad_channels: list of int
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
                 
    
        
class SpikeObj(object):
    
    htype = 'Spike'
    def __repr__(self):
        return str(self.tstamp)

    def __init__(self,waveform,tstamp,cluster,features):
        self.waveform = waveform        
        self.tstamp = tstamp
        self.cluster = cluster
        self.features = features
        
    def plot(self,ax = None, spines = ['left', 'bottom']):
        if ax == None:
            fig, ax = plt.subplots(1)
        ax.plot(self.waveform)
        adjust_spines(ax, spines)
        
class SpikeList(object):
    event = [] 
    def __addEvent__(self,obj):
        self.event.append(obj)
    def __removeEvent__(self,idx):
        del self.event[idx]
    def __repr__(self):
        return '%s events' % len(self.event)
    def __getcluster__(self):
        cluster = np.array([])
        for ev in self.event:
            cluster = np.append(cluster,ev.cluster)
        return cluster
        
    def __gettstamp__(self):
        tstamp = np.array([])
        for ev in self.event:
            tstamp = np.append(tstamp,ev.tstamp)
        return tstamp
        
    def __getfeatures__(self):
        features = np.array([])
        for ev in self.event:
            features = np.append(features,ev.features)
        return features
        
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
        
            



class HFOObj(object):
    def __repr__(self):
        return self.evtype

    def __init__(self,htype,ch,tstamp,other):
        self.htype = htype
        self.tstamp = tstamp
        self.ch = ch
        self.other = other