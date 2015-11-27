# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:14:46 2015

@author: anderson
"""

import numpy as np
import scipy.signal as sig

def pinknoise(N):
    '''
    Create a pink noise with N points.
    N - Number of samples to be returned
    '''
    M = N
    if N % 2:
        
        N += 1
    x = np.random.randn(N)
    
    X = np.fft.fft(x)
    
    nPts = int(N/2 + 1)
    n = range(1,nPts+1)
    n = np.sqrt(n)
    
    X[range(nPts)] = X[range(nPts)]/n
    X[range(nPts,N)] = np.real(X[range(N/2-1,0,-1)]) - 1j*np.imag(X[range(N/2-1,0,-1)])
    
    y = np.fft.ifft(X)
    
    y = np.real(y)
    
    y -= np.mean(y)
    y /= np.sqrt(np.mean(y**2))
    if M % 2 == 1:
        y = y[:-1]
    return y


    
def brownnoise(N):
    '''
    Create a brown noise with N points.
    N - Number of samples to be returned
    '''
    M = N
    if N % 2:
        
        N += 1
    x = np.random.randn(N)
    
    X = np.fft.fft(x)
    
    nPts = int(N/2 + 1)
    n = range(1,nPts+1)
    
    X[range(nPts)] = X[range(nPts)]/n
    X[range(nPts,N)] = np.real(X[range(N/2-1,0,-1)]) - 1j*np.imag(X[range(N/2-1,0,-1)])
    
    y = np.fft.ifft(X)
    
    y = np.real(y)
    
    y -= np.mean(y)
    y /= np.sqrt(np.mean(y**2))
    if M % 2 == 1:
        y = y[:-1]
    return y
    
    
def wavelet(numcycles,f,srate):
    '''
    Create a wavele
    numcycles - number of cycles (gaussian window)
    f - central frequency
    srate - signal sample rate
    '''
    N = float(2*srate*numcycles)/(f)
    time = np.linspace(-numcycles/float(f),numcycles/float(f),N)
    std = numcycles/(2*np.pi*f)
    wave = np.exp(2*1j*np.pi*f*time)*np.exp(-(time**2)/(2*(std**2)))
    return wave,time
    
def hfo(srate = 2000, f=None, numcycles = None):
    '''
    Create a HFO. 
    f = None (Default) - Create a random HFO with central frequency between 60-600 Hz.
    numcycles = None (Default) - Create a random HFO with numcycles between 9 - 14.
    '''
    if numcycles is None:
        numcycles = np.random.randint(9,15)
    if f is None:
        f = np.random.randint(60,600)
    wave,time = wavelet(numcycles,f,srate)
    return np.real(wave), time

def spike(srate = 2000, f=None, numcycles = None):
    '''
    Create a spike. 
    f = None (Default) - Create a random Spike with central frequency between 60-600 Hz.
    numcycles = None (Default) - Create a random Spike with numcycles between 1 - 2.
    '''
    if numcycles is None:
        numcycles = np.random.randint(1,3)
    if f is None:
        f = np.random.randint(60,600)
    wave,time = wavelet(numcycles,f,srate)
    return -np.real(wave),time
    
def find_max(data, thr=None):
    '''
    return the index of the local maximum
    '''
    value = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1
    if thr is not None:
        value = [x for x in value if data[x] > thr]
    return value 
    
def find_min(data, thr=None):
    '''
    return the index of the local minimum
    '''
    value = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1
    if thr is not None:
        value = [x for x in value if data[x] < thr]
    return value
    
def find_maxandmin(data, thr=None):
    '''
    return the index of the local maximum and minimum
    '''
    value = np.diff(np.sign(np.diff(data))).nonzero()[0] + 1
    if thr is not None:
        
        value = [x for x in value if abs(data[x]) > abs(thr)]
    return value 
    
    
def STD(env,ths):
    '''
    Calcule threshold by STD
    '''
    ths_value = np.mean(env) + ths*np.std(env)
    return ths_value

def Tukey(env,ths):
    '''
    Calcule threshold by Tukey
    '''
    ths_value = np.percentile(env,75) + ths*(np.percentile(env,75)-np.percentile(env,25))
    return ths_value

def percentile(env,ths):
    '''
    Calcule threshold by Percentile
    '''
    ths_value = np.percentile(env,ths)
    return ths_value

def Quian(env,ths):
    '''
    Calcule threshold by Quian
    Quian Quiroga, R. 2004. Neural Computation 16: 1661–87.
    '''
    ths_value = ths * np.median(np.abs(env)) / 0.6745
    return ths_value
    
    
def Hilbert_envelope(x):
    return np.abs(sig.hilbert(x))

def Hilbert_energy(x,window_size = 6):
    return np.abs(sig.hilbert(x))**2
    
def RMS(a, window_size = 6):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'same'))

    
def STenergy(a, window_size = 6):
    '''
    Calcule Short time energy - 
    Dümpelmann et al, 2012.  Clinical Neurophysiology: 123 (9): 1721–31.
    '''
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.convolve(a2, window, 'same')
    
    
def line_lenght(a, window_size = 6):
    '''
    Calcule Short time line leght - 
    Dümpelmann et al, 2012.  Clinical Neurophysiology: 123 (9): 1721–31.
    '''
    a2 = np.abs(np.diff(a))
    window = np.ones(window_size)/float(window_size)
    data =  np.convolve(a2, window, 'same')
    data = np.append(data,data[-1])
    return data
    
    
def whitening(data):
    data = np.diff(data)
    data = np.append(data,data[-1])
    return data
    
def filt(x,low_cut=60.,high_cut=600.,sample_rate=2000.,window = ('gaussian',10)):
    if high_cut is not None and low_cut is not None:
        f = [low_cut,high_cut]
        pass_zero=False
    elif high_cut is None:
        f = low_cut
        pass_zero=False
    if low_cut is None:
        f = high_cut       
        pass_zero=True
    
    nyq = sample_rate/2
    numtaps = int(sample_rate/10 + 1)
    b = sig.firwin(numtaps,f,pass_zero=pass_zero,window=window,nyq=nyq)
    filtered = sig.filtfilt(b,np.array([1]),sig.detrend(x))
    return filtered
    
    
def ishfo(filtered,x,ths,min_dur = 10., min_separation = 66.):
    
    subthsIX = np.asarray(np.nonzero(x < ths)[0])# subthreshold index
    subthsInterval = np.diff(subthsIX) # interval between subthreshold
    sIX = subthsInterval > min_dur # index of subthsIX bigger then minimal duration
    start_ix = subthsIX[sIX] + 1 # start index of events
    end_ix = start_ix + subthsInterval[sIX]-1 # end index of events
    to_remove = np.asarray(np.nonzero(start_ix[1:]-end_ix[0:-1] < min_separation)[0]) # find index of events separeted by less the minimal interval
    start_ix = np.delete(start_ix, to_remove+1) # removing
    end_ix = np.delete(end_ix, to_remove) #removing
    if start_ix.shape[0] != 0:
        locs = np.diff(np.sign(np.diff(filtered))).nonzero()[0] + 1 # local min+max
        to_remove = []
        for ii in range(start_ix.shape[0]):
            if np.nonzero((locs > start_ix[ii]) & (locs < end_ix[ii]))[0].shape[0] < 6:
                to_remove.append(ii)
        start_ix = np.delete(start_ix, to_remove) # removing
        end_ix = np.delete(end_ix, to_remove) #removing
        
    if start_ix.shape[0] != 0:
        return True
        if start_ix.shape[0]>1:
            print start_ix.shape[0]
    else:
        return False
        
        
def calc_stat(data,filtered,ths,start,end,True_ev,v=True):
    
    aux = []
    for s,e in zip(start,end):
        aux = np.append(aux,ishfo(filtered[int(s):int(e)],data[int(s):int(e)],ths))
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for hf,rs in zip(True_ev,aux):
        hf = bool(hf)
        if hf and rs:
            TP +=1
        elif hf and not rs:
            FN +=1
        elif rs and not hf:
            FP +=1
        elif not rs and not hf:
            TN +=1
    sens = TP*100./(TP+FN)
    spec = TN*100./(TN+FP)
    if v:
        print '                   HFO          '
        print '             True   |   False  |  Total'
        print '______________________________________'
        print '                    |          |'
        print ' detected     %3i   |    %3i   |   %3i' % (TP,FP,TP+FP)
        print '                    |          |'
        print '   not        %3i   |    %3i   |   %3i' % (FN,TN,TN+FN)
        print '______________________________________'
        print '  Total       %3i   |    %3i   |   %3i' % (TP+FN,TN+FP,TP+TN+FP+FN)
        print '\n sens(%%) = %3i , spec(%%) = %3i' % (sens,spec)
    return sens,spec