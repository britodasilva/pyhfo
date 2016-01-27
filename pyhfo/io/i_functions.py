# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:12:23 2015

@author: anderson
"""

import h5py
import numpy as np
import RHD
import scipy.io as sio
from pyhfo.core import DataObj, SpikeObj, hfoObj, EventList

    
def open_dataset(file_name,dataset_name,htype = 'auto'):
    '''
    open a dataset in a specific file_name
    
    Parameters
    ----------
    file_name: str 
        Name of the HDF5 (.h5) file 
    dataset_name: str
        Name of dataset to open
    htype: str, optional
        auto (the default) - read htype from HDF file 
        Data - DataObj type
        Spike - SpikeObj type
        hfo - hfoObj type
    '''
    # reading h5 file
    h5 = h5py.File(file_name,'r+')
    # loading dataset
    dataset = h5[dataset_name]
    # getting htype
    if htype == 'auto':
        htype = dataset.attrs['htype']      
    
    if htype == 'Data':
        # Sample Rate attribute
        sample_rate = dataset.attrs['SampleRate[Hz]']
        n_points         = dataset.shape[0]
        end_time         = n_points/sample_rate
        # Amplitude Unit
        if 'amp_unit' in dataset.attrs:
            amp_unit = dataset.attrs['amp_unit']
        else:
            amp_unit = 'AU'
        # Time vector
        if 'Time_vec_edge' in dataset.attrs:
            edge = dataset.attrs['Time_vec_edge']
            time_vec = np.linspace(edge[0],edge[1],n_points,endpoint=False)
        else:
            time_vec = np.linspace(0,end_time,n_points,endpoint=False)
        # Check if has 'Bad_channels' attribute, if not, create one empty
        if len([x for x in dataset.attrs.keys() if x == 'Bad_channels']) == 0:
            dataset.attrs.create("Bad_channels",[],dtype=int)
        # Load bad channels
        bad_channels = dataset.attrs["Bad_channels"]    
        # Creating dictionary
        Data = DataObj(dataset[:],sample_rate,amp_unit,dataset.attrs['Channel_Labels'],time_vec,bad_channels,file_name,dataset_name)
    
    elif htype == 'list':
        # Time vector
        keys  = dataset.keys()
        ch_labels = dataset.attrs['ch_labels']
        time_edge = dataset.attrs['time_edge']
        Data = EventList(ch_labels,time_edge,file_name,dataset_name)
        for k in keys:
            waveform =  dataset[k][:]
            tstamp = dataset[k].attrs['tstamp']
            evhtype = dataset[k].attrs['htype']
            channel = dataset[k].attrs['channel']
            if evhtype == 'Spike':
                clus = dataset[k].attrs['cluster']
                feat = dataset[k].attrs['features']
                time_edge = dataset[k].attrs['time_edge'] 
                spk = SpikeObj(channel,waveform,tstamp,clus,feat,time_edge)
                Data.__addEvent__(spk)
            elif evhtype == 'HFO':
                
                tstamp_idx = dataset[k].attrs['tstamp_idx'] 
                start_idx  = dataset[k].attrs['start_idx']
                end_idx  = dataset[k].attrs['end_idx']
                ths_value  = dataset[k].attrs['ths_value']
                sample_rate = dataset[k].attrs['sample_rate']
                cutoff = dataset[k].attrs['cutoff']
                info = dataset[k].attrs['info']
                hfo = hfoObj(channel,tstamp,tstamp_idx, waveform,start_idx,end_idx,ths_value,sample_rate,cutoff,info)
                Data.__addEvent__(hfo)
    
    h5.close()
    return Data
    

    
def loadRDH(filename):
    ''' 
    Created to load 64 channels at port A -  code need change if use diferent configuration. 
    It use RHD.py file to read and return DataObj
    
    Parameters
    ----------
    file_name: str 
        Name of the intran (.rhd) file 
    
    '''
    # load file
    myData = RHD.openRhd(filename)
    # get sample rate
    sample_rate = myData.sample_rate
    # get channels 
    myChannels = myData.channels
    # create a empty signal
    signal = np.zeros((myChannels['A-000'].getTrace().size,64))
    signal[:] = np.NAN
    labels = [] 
    for ch in range(64):
        if ch < 10:
            label = "A-00" + str(ch)
        else:
            label = "A-0" + str(ch)
        signal[:,ch] = myChannels[label].getTrace()
        labels.append(label)
    signal *= 0.195 # according the Intan, the output should be multiplied by 0.195 to be converted to micro-volts
    amp_unit = '$\mu V$'
    # Time vector   
    n_points  = signal.shape[0]
    end_time  = n_points/sample_rate
    time_vec  = np.linspace(0,end_time,n_points,endpoint=False)
    Data = DataObj(signal,sample_rate,amp_unit,labels,time_vec,[])
    return Data

def loadMAT(slice_filename,parameters_filename):
    '''
    Created to convert .mat files with specific configuration for ECoG data of Newcastle Hospitals and create a dict.
    If you want to load other .mat file, use scipy.io. loadmat and create_DataObj
    
    Parameters
    ----------
    slice_filename: str 
        Name of the slice (.mat) file 
    parameters_filename: str 
        Name of the parameters (.mat) file
    
    '''
    mat = sio.loadmat(parameters_filename, struct_as_record=False, squeeze_me=True)
    parameters = mat['parameters']
    ch_l = parameters.channels
    ch_labels = [str(x) for x in ch_l]
    sample_rate = parameters.sr
    f = sio.loadmat(slice_filename, struct_as_record=False, squeeze_me=True)
    Data = f['Data']
    time_vec = Data.time_vec
    signal = Data.raw.T
    amp_unit = '$\mu V$'
    Data = DataObj(signal,sample_rate,amp_unit,ch_labels,time_vec,[])
    return Data
    
    
def loadSPK_waveclus(filename,EventList,ch):
    '''
    load Spikes sorted by wave_clus.
    Parameters
    ----------
    filename: str
        Name of the spike (.mat) file 
    EventList: EventList
    
    '''
    
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    clusters = mat['cluster_class'][:,0]
    times = mat['cluster_class'][:,1]/1000
    spikes = mat['spikes']
    features = mat['inspk']
    labels = []
    for cl in range(int(max(clusters))+1):
        labels.append('Cluster '+str(cl))
    for idx,waveform in enumerate(spikes):
        tstamp = times[idx]
        clus = clusters[idx]
        feat= features[idx]
        time_edge = [-20,44]
        spk = SpikeObj(ch,waveform,tstamp,clus,feat,time_edge)
        EventList.__addEvent__(spk)
    return EventList
    

def get_info(folder):
    
    # load file
    print folder+'info.rhd'
    myData = RHD.openRhd(folder+'info.rhd')
    # get sample rate
    return myData
    
    
def loadDAT(filename,srate,label='data',amp=True):
    '''
    amp = True if it is from Amplifier, False if it is from ADC board
    
    '''
    fh = open(filename,'r')
    fh.seek(0)
    if amp:
        data = np.fromfile(fh, dtype=np.int16, count=-1)
    else:
        data = np.fromfile(fh, dtype=np.uint16, count=-1)
    fh.close()
    data = np.double(data)
    if amp:
        data *= 0.195 # according the Intan, the output should be multiplied by 0.195 to be converted to micro-volts
    else:
        data *= 0.000050354 # according the Intan, the output should be multiplied by 0.195 to be converted to micro-volts
    amp_unit = '$\mu V$'
    # Time vector   
    n_points  = data.shape[0]
    end_time  = n_points/srate
    time_vec  = np.linspace(0,end_time,n_points,endpoint=False)
    
    Data = DataObj(data,srate,amp_unit,label,time_vec,[])
    return Data
        
