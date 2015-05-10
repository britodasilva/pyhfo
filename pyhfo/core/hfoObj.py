# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:03:56 2015

@author: anderson
"""
from pyhfo.ui import plot_single_hfo
from .HFOSpectrum import HFOSpectrum

class hfoObj(object):
    def __repr__(self):
        return self.tstamp

    def __init__(self,channel,tstamp,tstamp_idx, waveform,start_idx,end_idx,ths_value,sample_rate,cutoff,info):
        self.htype = 'HFO'
        self.channel = channel              # channel  
        self.tstamp = tstamp                # Time stamp in sec. (center of event- max amp)
        self.tstamp_idx = tstamp_idx        # Time stamp in points. (center of event- max amp)
        self.waveform = waveform            # numpy array shape (sample_rate,2). 
                                            # first col - 1 second of raw wave
                                            # second col - 1 second of filtered wave
                                            # centered in tstamp (0.5 sec before and 0.5 after)
        self.start_idx = start_idx          # start index - when in waveform start HFO
        self.end_idx = end_idx              # end index - when in waveform end HFO
        self.ths_value = ths_value          # ths_value - value of choosen threshold
        self.sample_rate = sample_rate      # sample rate of recording
        self.cutoff = cutoff                # cutoff frequencies
        self.info = info                    # info about the method of detection (cuttofs, order)
        self.duration = (end_idx - start_idx + 1) * sample_rate # calculate the event duration
        self.spectrum = HFOSpectrum(self,cutoff) # spectrum object
        self.peak_amp = waveform[tstamp_idx,1] # get the peak amplitude value
        
        
    def plot(self,envelope = True, figure_size = (15,10),dpi=600):        
        plot_single_hfo(self, envelope = envelope, figure_size = figure_size,dpi=dpi)
        


   