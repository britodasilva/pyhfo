# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:03:27 2015

@author: anderson
"""

class SpikeObj(object):
    htype = 'Spike'
    def __repr__(self):
        return str(self.tstamp)

    def __init__(self,waveform,tstamp,cluster,features):
        self.waveform = waveform        
        self.tstamp = tstamp
        self.cluster = cluster
        self.features = features