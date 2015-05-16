# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:04:26 2015

@author: anderson
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from . import pop_channel, eegfilt

def phase_coupling(SPK,Data,low_cut,high_cut,ch=None,order = None,window = ('kaiser',0.5),nbins = 32):
    ''' 
    Caulculate the phase coupling in certain frequency band
    
    Parameters
    ----------
    SPK: SpkObj
    Data: DataObj
    low_cut: int,
        Low cut frequency
    high_cut: int,
        High cut frequency
        
    '''
    plt.figure(figsize=(10,10))
    if len(Data.data.shape) == 1:
       signal = Data
    elif ch == None:
        raise Exception('Choose a ch')
    else:
        signal = pop_channel(Data,ch)   
    filt = eegfilt(signal,low_cut,high_cut,order = order,window = window)
    phs = np.angle(sig.hilbert(filt.data))
    #ang = np.linspace(0.0,2*np.pi,nbins, endpoint=False)
    loc = [int(x) for x in SPK.__getlist__('tstamp')*signal.sample_rate]
    k = phs[loc]
    plt.subplot(121,polar=True)
    plt.hist(k,bins=nbins)
    plt.yticks([])
    plt.subplot(122,polar=False)
    nbin = np.concatenate((k+np.pi,k+3*np.pi))
    plt.hist(nbin,bins=2*nbins)
    plt.xticks(np.linspace(0.0,4*np.pi,9),[str(int(x)) for x in 180*np.linspace(0.0,4*np.pi,9)/np.pi])
    plt.xlim([0,4*np.pi])
    return k