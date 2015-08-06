# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:04:03 2015

@author: anderson
"""

from pyhfo.core import eegfilt, hfoObj, EventList
import numpy as np
import math
import scipy.signal as sig
from IPython.parallel import Client
import itertools

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
                        ths = 5, ths_method = 'STD', min_dur = 3, min_separation = 2, energy = False):
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
    if nch == 1:
        print 'Finding in channel'
        filt = filtOBj.data
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
            waveform[:,0] = Data.data[Lindex]
            waveform[:,1] = filtOBj.data[Lindex]
            start_idx = np.nonzero(Lindex==s)[0][0]
            end_idx = np.nonzero(Lindex==e)[0][0]
            hfo = hfoObj(0,tstamp,tstamp_idx, waveform,start_idx,end_idx,ths_value,sample_rate,cutoff,info)
            HFOs.__addEvent__(hfo)
    else:
        for ch in range(nch):
            if ch not in filtOBj.bad_channels:
                print 'Finding in channel ' + filtOBj.ch_labels[ch]
                filt = filtOBj.data[:,ch]
                if energy:
                    env  = np.abs(sig.hilbert(filt))**2
                else:
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



def findHFO_filtbank(Data,low_cut = 50,high_cut= None, ths = 5, max_ths = 10,par = False):
    '''
    Find HFO by Filter-bank method.
    by Anderson Brito da Silva - 29/jul/2015
    
    
    Parameters
    ----------
    Data: DataObj
        Data object to filt/find HFO
    low_cut: int
        50 (default) - Low cut frequency in Hz. 
    high_cut: int
        High cut frequency in Hz. If None, high_cut = nyrqst
    ths : int, optional
        3 (default) - threshold for z-score 
    max_ths : int, optional
        20 (default) - max threshold for z-score
   
    '''
    import sys
    
    def create_wavelet(f,time):
        import numpy as np
        numcycles = 13
        std = numcycles/(2*np.pi*f)
        wavelet = np.exp(2*1j*np.pi*f*time)*np.exp(-(time**2)/(2*(std**2)))
        wavelet /= max(wavelet)
        return wavelet


    
    def filt_wavelet(data_ch,wavelet): 
        import scipy.signal as sig
        x = sig.fftconvolve(data_ch,wavelet,'same')
        return x

    
    def bin_filt(filt):
        import numpy as np
        ab_x = abs((filt-np.mean(filt[200:-200]))/np.std(filt[200:-200]))
        bin_x = np.array([1 if y < 5 or y > 10 else 0 for y in ab_x])
        return bin_x
    
    
    def find_min_duration(f,sample_rate):
        import math
        # Transform min_dur from cicles to poinst - minimal duration of HFO (Default is 3 cicles)
        min_dur = 3
        min_dur = math.ceil(min_dur*sample_rate/f)
        return min_dur
        
    def find_min_separation(f,sample_rate):
        import math
        # Transform min_separation from cicles to points - minimal separation between events
        min_separation = 2 
        min_separation = math.ceil(min_separation*sample_rate/f)
        return min_separation
        
    def find_start_end(x, bin_x,min_dur,min_separation,max_local=True):
        import numpy as np
        
        subthsIX = bin_x.nonzero()[0] # subthreshold index
        subthsInterval = np.diff(subthsIX) # interval between subthreshold
        
        sIX = subthsInterval > min_dur # index of subthsIX bigger then minimal duration
        start_ix = subthsIX[sIX] + 1 # start index of events
        end_ix = start_ix + subthsInterval[sIX]-1 # end index of events
        
        to_remove = np.asarray(np.nonzero(start_ix[1:]-end_ix[0:-1] < min_separation)[0]) # find index of events separeted by less the minimal interval
        start_ix = np.delete(start_ix, to_remove+1) # removing
        end_ix = np.delete(end_ix, to_remove) #removing
        if max_local:
            if start_ix.shape[0] != 0:
                locs = np.diff(np.sign(np.diff(x))).nonzero()[0] + 1 # local min+max
                to_remove = []
                for ii in range(start_ix.shape[0]):
                    if np.nonzero((locs > start_ix[ii]) & (locs < end_ix[ii]))[0].shape[0] < 6:
                        to_remove.append(ii)
                start_ix = np.delete(start_ix, to_remove) # removing
                end_ix = np.delete(end_ix, to_remove) #removing
        
        return start_ix, end_ix
        
    def se_to_array(arrlen,se):
        import numpy as np
        z = np.zeros((arrlen,1))
        for ii in range(se[0].shape[0]):
            z[se[0][ii]:se[1][ii],0] = 1
            
        return z
    
    
    
    print 'Finding HFO by Wavelet Filter Bank'
    sys.stdout.flush()
    
        
    if low_cut == None and high_cut == None:
        raise Exception('You should determine the cutting frequencies') 
    sample_rate = Data.sample_rate
    # if no high cut, =nyrqst 
    if high_cut == None:
        high_cut = sample_rate/2
        
    cutoff = [low_cut,high_cut] # define cutoff
    noffilters = len(range(cutoff[0],cutoff[1],5)) # number of filters
    seg_len = 400. # milisecond
    npoints = seg_len*sample_rate/1000 # number of points of wavelet
    time = np.linspace(-seg_len/2000,seg_len/2000,npoints) #time_vec
    
    HFOs = EventList(Data.ch_labels,(Data.time_vec[0],Data.time_vec[-1]))
    info = str(low_cut) + '-' + str(high_cut) + ' Hz Wavelet Filter Bank'  
    if par:
        print 'Using Parallel processing',

        c = Client()
        dview  = c[:]
        print str(len(c.ids)) + ' cores'
        min_durs = dview.map_sync(find_min_duration,range(cutoff[0],cutoff[1],5),itertools.repeat(sample_rate,noffilters))
        print 'Durations',
       
        min_seps = dview.map_sync(find_min_separation,range(cutoff[0],cutoff[1],5),itertools.repeat(sample_rate,noffilters))
        print '/ Separations',
        sys.stdout.flush()
        wavelets = dview.map_sync(create_wavelet,range(cutoff[0],cutoff[1],5),itertools.repeat(time,noffilters))
        print '/ Wavelets'
        sys.stdout.flush()
        
    nch = Data.n_channels   
    for ch in range(nch):
        if ch not in Data.bad_channels:
            print 'Finding in channel ' + Data.ch_labels[ch]
            sys.stdout.flush()
            data_ch = Data.data[:,ch]
            arrlen = data_ch.shape[0]
            zsc = np.zeros((arrlen,noffilters))
            spect = np.zeros((arrlen,noffilters), dtype='complex' )
            if par:  
                
                sys.stdout.flush()
                filt_waves = dview.map_sync(filt_wavelet,itertools.repeat(data_ch,noffilters),wavelets)
                print 'Convolved',
                sys.stdout.flush()
                spect = np.array(filt_waves)
                bin_xs= dview.map_sync(bin_filt,filt_waves)
                print '/ Binarised',
                sys.stdout.flush()
                se = dview.map_sync(find_start_end,filt_waves,bin_xs,min_durs,min_seps)
                filt_waves = None
                bin_xs = None
                print '/ Found',
                sys.stdout.flush()
                z_list = dview.map_sync(se_to_array,itertools.repeat(arrlen,noffilters),se)
                zsc = np.squeeze(z_list)
                upIX = np.unique(np.nonzero(zsc==1)[1])
                other = np.ones(data_ch.shape)
                other[upIX] = 0
                print '/ Finalizing'
                sys.stdout.flush()
                start_ix, end_ix = find_start_end([],other,find_min_duration(cutoff[1],sample_rate),find_min_separation(cutoff[0],sample_rate),max_local=False)
                
                
            else:
                wavelets = map(create_wavelet,range(cutoff[0],cutoff[1],5),itertools.repeat(time,noffilters))
                

            
            for s, e in zip(start_ix, end_ix):
                index = np.arange(s,e)
                s_o = zsc[:,index]
                aux = spect[np.unique(s_o.nonzero()[0]),:]
                z = aux[:,index]
                HFOwaveform = np.mean(z,0)
                tstamp_points = s + np.argmax(HFOwaveform)
                tstamp = Data.time_vec[tstamp_points]
                s_o = None
                index = None
                HFOwaveform = None
                if tstamp_points-int(sample_rate/2) < 0 or tstamp_points+int(sample_rate/2)+1 > zsc.shape[1]:
                    continue
                
                Lindex = np.arange(tstamp_points-int(sample_rate/2),tstamp_points+int(sample_rate/2)+1)
                tstamp_idx = np.nonzero(Lindex==tstamp_points)[0][0]
                waveform = np.empty((Lindex.shape[0],2))
                waveform[:] = np.NAN
                waveform[:,0] = Data.data[Lindex,ch]
                

                s_o = zsc[:,Lindex]
                aux = spect[np.unique(s_o.nonzero()[0]),:]
                z = aux[:,Lindex]
                waveform[:,1] = np.mean(z,0)
                start_idx = np.nonzero(Lindex==s)[0][0]
                end_idx = np.nonzero(Lindex==e)[0][0]
                hfo = hfoObj(ch,tstamp,tstamp_idx, waveform,start_idx,end_idx,ths,sample_rate,cutoff,info)
                HFOs.__addEvent__(hfo)
                s_o = None
                Lindex = None
            print HFOs
            sys.stdout.flush()
    return HFOs
    
    