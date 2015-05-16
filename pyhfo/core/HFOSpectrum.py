# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:05:15 2015

@author: anderson
"""

import scipy.signal as sig
import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt


class HFOSpectrum(object):
    def __init__(self,hfoObj,cutoff):
        signal = sig.detrend(hfoObj.waveform[hfoObj.start_idx:hfoObj.end_idx,0]) # detrending
        next2power = 2**(hfoObj.sample_rate-1).bit_length() # next power of two of sample rate (power of 2 which contains at least 1 seg)
        signal = np.lib.pad(signal, int((next2power-len(signal))/2), 'constant', constant_values=0)
        self.F, self.Pxx = sig.welch(np.diff(signal), fs = hfoObj.sample_rate, nperseg = np.diff(signal).shape[-1])
        self.nPxx = self.Pxx/np.sum(self.Pxx)
        self.entropy = stat.entropy(self.nPxx)/(np.log(len(self.nPxx))/np.log(np.e))
        self.power_index, self.peak_freq, self.peak_win_power = self.peak_power(cutoff[0],cutoff[1],npoints = 40,normalised = True, plot = False, v = False)
    
    def plot(self, normalised = True, cutoff = None, v = True, ax = None):
        if ax == None:
            f = plt.figure()    
            ax = f.add_subplot(111)
        if cutoff == None:
            if normalised:
                ax.plot(self.F,self.nPxx)
            else:
                ax.plot(self.F,self.Pxx)
        else:
            self.peak_power(cutoff[0],cutoff[1],npoints = 40,normalised = normalised, plot = True, v = v, ax = ax)
            
            
    
    def peak_power(self,low_cut,high_cut,npoints = 40,normalised = True, plot = True, v = True, ax = None):
        '''
        Find peak in the power spectrum between the cutoff frequencies and calculate the
        undercurve area.
        
        Parameters
        ----------
        low_cut: int
            Low cutoff edge
        high_cut: int
            High cutoff edge
        npoints: int
            40 (Default) - number of points to calculate around peak power. By default, each point = 1 Hz, so calculate a 40 Hz window.
        normalised: boolean
            True (Default) - Make the power in normalised spectrum
        plot: boolean
            True (Default) - Plot results
        '''
        if normalised:
            power = self.nPxx
        else:
            power = self.Pxx
        event_band = np.nonzero((self.F>low_cut) & (self.F<= high_cut))[0] # selecting the band indexs
        max_e_idx = np.argmax(power[event_band]) # peak frequency idx
        # selecting a 40 points windows between peak (40 Hz)
        if max_e_idx < (npoints/2)-1:
            win_e = np.arange(npoints)
        elif max_e_idx + (npoints/2) > event_band.shape[0]:
            win_e = np.arange(event_band.shape[0]-(npoints/2),event_band.shape[0])
        else:
            win_e = np.arange(max_e_idx-(npoints/2),max_e_idx+(npoints/2))
        if plot:
            if ax == None:
                f = plt.figure()    
                ax = f.add_subplot(111)
            self.plot(normalised=normalised,ax = ax)
            ax.plot(self.F[event_band][max_e_idx], power[event_band][max_e_idx],'o')
            ax.fill_between(self.F[event_band], power[event_band])
            ax.fill_between(self.F[event_band][win_e], power[event_band][win_e],facecolor='y')
        
                
        band_power = sum(power[event_band])
        power_around_peak = sum(power[event_band][win_e]) # under curve area
        peak_freq   =  self.F[event_band][max_e_idx]
        if v:
            print 'Power Band: ' + str(band_power) + '; Peak Frequency: ' + str(peak_freq) + '; Power window: ' + str(power_around_peak)
        return band_power, peak_freq, power_around_peak