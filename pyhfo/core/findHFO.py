# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:04:03 2015

@author: anderson
"""

from pyhfo.core import eegfilt, hfoObj, EventList
import numpy as np
import math
import scipy.signal as sig
import itertools
import h5py



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
                        ths = 5, ths_method = 'STD', min_dur = 3, min_separation = 2, energy = False,
                        whitening = True,filter_test=False,rc = None ,dview = None):
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
    import sys
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
    if rc is not None:
        print 'Using Parallel processing',
        print str(len(rc.ids)) + ' cores'
        sys.stdout.flush()
        par = True
    if filter_test:
        filtOBj = eegfilt(Data,low_cut, high_cut,order,window,whitening,filter_test)
        return
    else:
        filtOBj = eegfilt(Data,low_cut, high_cut,order,window,whitening,rc = rc, dview = dview)
    nch = filtOBj.n_channels
    if order == None:
        order = int(sample_rate/10)
    info = str(low_cut) + '-' + str(high_cut) + ' Hz filtering; order: ' + str(order) + ', window: ' + str(window) + ' ; ' + str(ths) + '*' + ths_method + '; min_dur = ' + str(min_dur) + '; min_separation = ' + str(min_separation) + '; whiteting = ' + str(whitening)
    print info
    print filtOBj.data.shape   
    sys.stdout.flush()
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



def findHFO_filtbank(Data,low_cut = 50,high_cut= None, ths = 5, max_ths = 10,par = False, save = None, replace = False, exclude = [],rc = None ,dview = None):
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
    import time
    
    def clear_cache(rc,dview):
        rc.purge_results('all')
        rc.results.clear()
        rc.metadata.clear()
        dview.results.clear()
        assert not rc.outstanding
        rc.history = []
        dview.history = []
        
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
        #bin_x = np.array([1 if y < 5 or y > 10 else 0 for y in abs((filt-np.mean(filt[200:-200]))/np.std(filt[200:-200]))])
        #bin_x = np.zeros(filt.shape)
        q75, q25 = np.percentile(np.abs(filt[200:-200]), [75 ,25])
        iqr = q75 - q25
        #bin_x[np.nonzero(filt.real<(q75+3*iqr))] = 1        
        bin_x = np.array([1 if y < (q75+3*iqr) else 0 for y in np.abs(filt)])        
        return bin_x
    
    
    def find_min_duration(f,sample_rate):
        import math
        # Transform min_dur from cicles to poinst - minimal duration of HFO (Default is 3 cicles)
        min_dur = math.ceil(3*sample_rate/f)
        return min_dur
        
    def find_min_separation(f,sample_rate):
        import math
        # Transform min_separation from cicles to points - minimal separation between events
        min_separation = math.ceil(2*sample_rate/f)
        return min_separation
        
    def find_start_end(x, bin_x,min_dur,min_separation,max_local=True):
        import numpy as np

        
        subthsIX = bin_x.nonzero()[0] # subthreshold index
        
        subthsInterval = np.diff(subthsIX) # interval between subthreshold
        
        sIX = subthsInterval > min_dur # index of subthsIX bigger then minimal duration
        start_ix = subthsIX[sIX] + 1 # start index of events
        end_ix = start_ix + subthsInterval[sIX]-1 # end index of events
        
       
        to_remove = np.array(np.nonzero(start_ix[1:]-end_ix[0:-1] < min_separation)[0]) # find index of events separeted by less the minimal interval
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
    scales = np.logspace(np.log(cutoff[0]),np.log(cutoff[1]), base=np.e,num = 30, endpoint=False)
    noffilters = len(scales) # number of filters
    seg_len = 400. # milisecond
    npoints = seg_len*sample_rate/1000 # number of points of wavelet
    time_vec = np.linspace(-seg_len/2000,seg_len/2000,npoints) #time_vec
    
    
    
    if save is not None:
        file_name = save[0]
        obj_name = save[1]
        save_opt = True
        HFOs = EventList(Data.ch_labels,(Data.time_vec[0],Data.time_vec[-1]),file_name = file_name, dataset_name = obj_name)
        # open or creating file 
        h5 = h5py.File(file_name,'a')
        #deleting previous dataset
        
        if obj_name in h5:
            if replace:
                del h5[obj_name]
                group = h5.create_group(obj_name)
                group.attrs.create('htype',HFOs.htype)
                group.attrs.create('time_edge',[HFOs.time_edge[0],HFOs.time_edge[-1]])
                group.attrs.create('ch_labels', HFOs.ch_labels[:])
                ev_count = 0
                print 'Created Dataset ' + file_name + ' ' + obj_name
            else:
                group = h5[obj_name]
                ev_count = len(group.items())
                print 'Open Dataset ' + file_name + ' ' + obj_name 
        else:
            group = h5.create_group(obj_name)
            group.attrs.create('htype',HFOs.htype)
            group.attrs.create('time_edge',[HFOs.time_edge[0],HFOs.time_edge[-1]])
            group.attrs.create('ch_labels', HFOs.ch_labels[:])
            ev_count = 0
            print 'Created Dataset ' + file_name + ' ' + obj_name
            
                
    else:
        save_opt = False
        HFOs = EventList(Data.ch_labels,(Data.time_vec[0],Data.time_vec[-1]))
        
    info = str(low_cut) + '-' + str(high_cut) + ' Hz Wavelet Filter Bank'  
    if par:
        print 'Using Parallel processing',
        
        print str(len(rc.ids)) + ' cores'
        min_durs = map(find_min_duration,scales,itertools.repeat(sample_rate,noffilters))
        print 'Durations',
       
        min_seps = map(find_min_separation,scales,itertools.repeat(sample_rate,noffilters))
        print '/ Separations',
        sys.stdout.flush()
        wavelets = map(create_wavelet,scales,itertools.repeat(time_vec,noffilters))
        print '/ Wavelets'
        sys.stdout.flush()

    nch = Data.n_channels   
    for ch in [x for x in range(nch) if not x in exclude]:
        if ch not in Data.bad_channels:
            btime = time.time()
            if save_opt:
                del HFOs
                h5.close()
                h5 = h5py.File(file_name,'a')
                group = h5[obj_name]
                ev_count = len(group.items())
                print group, ev_count
                  
                HFOs = EventList(Data.ch_labels,(Data.time_vec[0],Data.time_vec[-1]),file_name = file_name, dataset_name = obj_name)
            print 'Finding in channel ' + Data.ch_labels[ch]
            sys.stdout.flush()
            
            arrlen = Data.data[:,ch].shape[0]
            zsc = np.zeros((arrlen,noffilters))
            spect = np.zeros((arrlen,noffilters), dtype='complex' )
            if par:  
                
                sys.stdout.flush()
                filt_waves = dview.map_sync(filt_wavelet,itertools.repeat(Data.data[:,ch],noffilters),wavelets)
                clear_cache(rc,dview)
                print 'Convolved',
                sys.stdout.flush()
                spect = np.array(filt_waves)
                bin_xs= dview.map_sync(bin_filt,filt_waves)
                clear_cache(rc,dview)
                print '/ Binarised',
                sys.stdout.flush()
                se = dview.map_sync(find_start_end,filt_waves,bin_xs,min_durs,min_seps)
                clear_cache(rc,dview)
                #print 'here'
                #_vars = sys.modules[__name__]
                #delattr(_vars, filt_waves)
                #delattr(_vars, bin_xs)
                print '/ Found',
                sys.stdout.flush()
                zsc = np.squeeze(dview.map_sync(se_to_array,itertools.repeat(arrlen,noffilters),se))
                clear_cache(rc,dview)
                upIX = np.unique(np.nonzero(zsc==1)[1])
                other = np.ones(Data.data[:,ch].shape)
                other[upIX] = 0
                print '/ Start-End'
                sys.stdout.flush()
                start_ix, end_ix = find_start_end([],other,find_min_duration(cutoff[1],sample_rate),find_min_separation(cutoff[0],sample_rate),max_local=False)
                
                
            else:
                wavelets = map(create_wavelet,scales,itertools.repeat(time_vec,noffilters))
                

            print 'Creating list',
            sys.stdout.flush()
            for s, e in zip(start_ix, end_ix):

                HFOwaveform = np.argmax(np.mean(spect[np.unique(zsc[:,np.arange(s,e)].nonzero()[0]),:][:,np.arange(s,e)],0))
                tstamp_points = s + HFOwaveform
                if HFOwaveform > int(sample_rate/2):
                    continue
                if tstamp_points+int(sample_rate/2)+1 < e:
                    continue
                if tstamp_points-int(sample_rate/2) < 0 or tstamp_points+int(sample_rate/2)+1 > zsc.shape[1]:
                    continue
   
     
                waveform = np.empty((np.arange(tstamp_points-int(sample_rate/2),tstamp_points+int(sample_rate/2)+1).shape[0],2))
                waveform[:] = np.NAN
                waveform[:,0] = Data.data[np.arange(tstamp_points-int(sample_rate/2),tstamp_points+int(sample_rate/2)+1),ch]
                

                waveform[:,1] = np.mean(spect[np.unique(zsc[:,np.arange(tstamp_points-int(sample_rate/2),tstamp_points+int(sample_rate/2)+1)].nonzero()[0]),:][:,np.arange(tstamp_points-int(sample_rate/2),tstamp_points+int(sample_rate/2)+1)],0)
                start_idx = np.nonzero(np.arange(tstamp_points-int(sample_rate/2),tstamp_points+int(sample_rate/2)+1)==s)[0][0]
                end_idx = np.nonzero(np.arange(tstamp_points-int(sample_rate/2),tstamp_points+int(sample_rate/2)+1)==e)[0][0]
                hfo = hfoObj(ch,Data.time_vec[tstamp_points],np.nonzero(np.arange(tstamp_points-int(sample_rate/2),tstamp_points+int(sample_rate/2)+1)==tstamp_points)[0][0], waveform,start_idx,end_idx,ths,sample_rate,cutoff,info)
                #if hfo.spectrum.peak_freq > low_cut or hfo.spectrum.peak_freq < high_cut:
                HFOs.__addEvent__(hfo)
                print '.',
                sys.stdout.flush()
                
            print '\n'
            print HFOs
            if save_opt:
                print '... Saving ...'
                sys.stdout.flush()
                for idx, ev in enumerate(HFOs.event):
                    name = ev.htype + '_' + str(idx+ev_count)
                    dataset  = group.create_dataset(name,data=ev.waveform)
                    dataset.attrs.create('htype', ev.htype)
                    dataset.attrs.create('tstamp', ev.tstamp)
                    dataset.attrs.create('channel', ev.channel)
                    dataset.attrs.create('tstamp_idx', ev.tstamp_idx)
                    dataset.attrs.create('start_idx', ev.start_idx)
                    dataset.attrs.create('end_idx', ev.end_idx)
                    dataset.attrs.create('ths_value', ev.ths_value)
                    dataset.attrs.create('sample_rate', ev.sample_rate)
                    dataset.attrs.create('cutoff', ev.cutoff)
                    dataset.attrs.create('info', ev.info)
            
            print 'Elapsed: %s' % (time.time() - btime)
                
                    
    h5.close()
    
               
                   
            

    
    