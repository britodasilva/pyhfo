# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:04:26 2015

@author: anderson
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from . import pop_channel, eegfilt
from pyhfo.ui import adjust_spines

def phase_coupling(SPK,Data,low_cut,high_cut,cluster = 'all', ch=None,order = None,window = ('kaiser',0.5),nbins = 32,color=None, saveplot = None):
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
    if color ==None:
            colours = ['#000000','#0000FF','#FF0000','#8fbc8f','yellow']
    if cluster == 'all':
        l = int(np.max(SPK.__getlist__('cluster'))+1)
             
    elif cluster == None:
        loc = [int(x*signal.sample_rate) for x in SPK.__getlist__('tstamp')]
        l = 1
    else:
        idx = [int(i) for i,x in enumerate(SPK.__getlist__('cluster')) if x== cluster]
        loc = [int(x*signal.sample_rate) for i,x in enumerate(SPK.__getlist__('tstamp')) if i in idx]
        l = 1
    
    c = 0
    for clus in range(l): 
        if l == 1:
            k = phs[loc]
        else:
            idx = [int(i) for i,x in enumerate(SPK.__getlist__('cluster')) if x== clus]
            loc = [int(x*signal.sample_rate) for i,x in enumerate(SPK.__getlist__('tstamp')) if i in idx]
            k = phs[loc]
        
        c += 1
        plt.subplot(l,2,c,polar=True)
        plt.hist(k,bins=nbins,facecolor=colours[clus])
        plt.yticks([])
        
        c += 1
        ax = plt.subplot(l,2,c,polar=False)
        nbin = np.concatenate((k,k+2*np.pi))
        plt.hist(nbin,bins=2*nbins,facecolor=colours[clus])
        plt.xticks(np.linspace(-np.pi,3*np.pi,9),[str(int(x)) for x in 180*np.linspace(-np.pi,3*np.pi,9)/np.pi])
        plt.xlim([-np.pi,3*np.pi])
        adjust_spines(ax, ['left','bottom'])
        
        r = np.sum(np.exp(1j*k))
        r = np.abs(r)/k.shape[0]
        print 'Cluster ' + str(clus) + ' r: ' + str(r)
        if saveplot != None:
            if type(saveplot) == str: 
                plt.savefig(saveplot, bbox_inches='tight')
            else:
                raise Exception('saveplot should be a string')
