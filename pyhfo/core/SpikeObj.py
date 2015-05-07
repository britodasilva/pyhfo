# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:03:27 2015

@author: anderson
"""
from pyhfo.ui import plot_single_spk

class SpikeObj(object):
   
    def __repr__(self):
        return str(self.tstamp)

    def __init__(self,waveform,tstamp,cluster,features):
        self.htype = 'Spike'
        self.waveform = waveform        
        self.tstamp = tstamp
        self.cluster = cluster
        self.features = features

        
    def plot(self,subplot = None, spines = ['left', 'bottom'],
             figure_size = (5,5),dpi=600,**kwargs):
        plot_single_spk(self,subplot, spines,figure_size,dpi,**kwargs)
        