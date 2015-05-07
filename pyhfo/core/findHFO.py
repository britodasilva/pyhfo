# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:04:03 2015

@author: anderson
"""

from pyhfo.core import eegfilt, hfoObj, EventList
import numpy as np
import math
import scipy.signal as sig

def findStartEnd(filt,env,ths,min_dur,min_separation):
            subthsIX = np.asarray(np.nonzero(env < ths)[0])# subthreshold index
            subthsInterval = np.diff(subthsIX) # interval between subthreshold
            sIX = subthsInterval > min_dur # index of subthsIX bigger then minimal duration
            start_ix = subthsIX[sIX] + 1 # start index of events
            end_ix = start_ix + subthsInterval[sIX]-1 # end index of events
            to_remove = np.asarray(np.nonzero(start_ix[1:]-end_ix[0:-1] < min_separation)[0]) # find index of events separeted by less the minimal interval
            start_ix = np.delete(start_ix, to_remove+1) # removing
            end_ix = np.delete(end_ix, to_remove) #removing
            if start_ix.shape[0] != 0:
                locs = np.diff(np.sign(np.diff(filt))).nonzero()[0] + 1 # local min+max
                to_remove = []
                for ii in range(start_ix.shape[0]):
                    if np.nonzero((locs > start_ix[ii]) & (locs < end_ix[ii]))[0].shape[0] < 6:
                        to_remove.append(ii)
                start_ix = np.delete(start_ix, to_remove) # removing
                end_ix = np.delete(end_ix, to_remove) #removing
            return (start_ix, end_ix)

def findHFO_filtHilbert(Data,low_cut,high_cut= None, order = None,window = ('kaiser',0.5),
                        ths = 5, ths_method = 'STD', min_dur = 3, min_separation = 2):
    '''
    Find HFO by Filter-Hilbert method.
    
    Parameters
    ----------
    Data: DataObj
        Data object to filt/find HFO
    low_cut: int
        Low cut frequency. 
    high_cut: int
        High cut frequency. If None, high_cut = nyrqst
    order: int, optional
        None (default) - Order of the filter calculated as 1/10 of sample rate
    window : string or tuple of string and parameter values
        Desired window to use. See `scipy.signal.get_window` for a list
        of windows and required parameters.
    ths : int, optional
        5 (default) - times value of threshold (5*STD for example) 
    ths_method: str, optional
        'STD' - Standard desviation above the mean
        'Tukey' - Interquartil interval above percentile 75
    min_dur: int, optional
        3 (default) - minimal number of cicle that event should last. Calculeted 
        the number of points that event should last by formula ceil(min_dur*sample_rate/high_cut)
    min_separation: int, optional
        2 (defalt) - minimal number of cicle that separete events. Calculetad 
        the number of points that separete events by formula ceil(min_separation*sample_rate/low_cut)
    '''
    if low_cut == None and high_cut == None:
        raise Exception('You should determine the cutting frequencies') 
    sample_rate = Data.sample_rate
    # if no high cut, =nyrqst 
    if high_cut == None:
        high_cut = sample_rate/2
        
    cutoff = [low_cut,high_cut]
    # Transform min_dur from cicles to poinst - minimal duration of HFO (Default is 3 cicles)
    min_dur = math.ceil(min_dur*sample_rate/high_cut)
    # Transform min_separation from cicles to points - minimal separation between events
    min_separation = math.ceil(min_separation*sample_rate/low_cut)
    # filtering
    filtOBj = eegfilt(Data,low_cut, high_cut,order,window)
    nch = filtOBj.n_channels
    if order == None:
        order = int(sample_rate/10)
    info = str(low_cut) + '-' + str(high_cut) + ' Hz filtering; order: ' + str(order) + ', window: ' + str(window) + ' ; ' + str(ths) + '*' + ths_method + '; min_dur = ' + str(min_dur) + '; min_separation = ' + str(min_separation) 
    HFOs = EventList(Data.ch_labels,(Data.time_vec[0],Data.time_vec[-1])) 
    for ch in range(nch):
        if ch not in filtOBj.bad_channels:
            print 'Finding in channel ' + filtOBj.ch_labels[ch]
            filt = filtOBj.data[:,ch]
            env  = np.abs(sig.hilbert(filt))
            if ths_method == 'STD':
                ths_value = np.mean(env) + ths*np.std(env)
            elif ths_method == 'Tukey':
                ths_value = np.percentile(env,75) + ths*(np.percentile(env,75)-np.percentile(env,25))
            start, end = findStartEnd(filt,env,ths_value,min_dur,min_separation)
            for s, e in zip(start, end):
                index = np.arange(s,e)
                HFOwaveform = env[index]
                tstamp_points = s + np.argmax(HFOwaveform)
                tstamp = Data.time_vec[tstamp_points]
                Lindex = np.arange(tstamp_points-int(sample_rate/2),tstamp_points+int(sample_rate/2)+1)
                
                tstamp_idx = np.nonzero(Lindex==tstamp_points)[0][0]
                waveform = np.empty((Lindex.shape[0],2))
                waveform[:] = np.NAN
                waveform[:,0] = Data.data[Lindex,ch]
                waveform[:,1] = filtOBj.data[Lindex,ch]
                start_idx = np.nonzero(Lindex==s)[0][0]
                end_idx = np.nonzero(Lindex==e)[0][0]
                hfo = hfoObj(ch,tstamp,tstamp_idx, waveform,start_idx,end_idx,ths_value,sample_rate,cutoff,info)
                HFOs.__addEvent__(hfo)
    return HFOs
