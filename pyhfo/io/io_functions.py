# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:12:23 2015

@author: anderson
"""

import h5py
import numpy as np
import RHD
import scipy.io as sio
from pyhfo.core import create_DataObj, SpikeObj, SpikeList

    
def open_dataset(file_name,dataset_name):
    '''
    open a dataset in a specific file_name, return DataObj
    '''
    # reading h5 file
    h5 = h5py.File(file_name,'r+')
    # loading dataset
    dataset = h5[dataset_name]
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
    DataObj = create_DataObj(dataset[:],sample_rate,amp_unit,dataset.attrs['Channel_Labels'],time_vec,bad_channels)
    h5.close()
    return DataObj
    
def save_dataset(DataObj,file_name,dataset_name):
    '''
    save DataObj in a dataset in a specific file_name
    '''
    h5 = h5py.File(file_name,'a')
    if dataset_name in h5:
        del h5[dataset_name]
    dataset  = h5.create_dataset(dataset_name,data=DataObj.data)
    dataset.attrs.create('SampleRate[Hz]',DataObj.sample_rate)
    dataset.attrs.create('amp_unit',DataObj.amp_unit)
    dataset.attrs.create('Bad_channels',DataObj.bad_channels)
    dataset.attrs.create('Channel_Labels', DataObj.ch_labels)
    dataset.attrs.create('Time_vec_edge',[DataObj.time_vec[0],DataObj.time_vec[-1]])
    h5.close()
    
def loadRDH(filename):
    ''' 
    Created to load 64 channels at port A -  code need change if use diferent configuration. 
    It use RHD.py file to read and return Data_dict
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
    DataObj = create_DataObj(signal,sample_rate,amp_unit,labels,time_vec,[])
    return DataObj
    

def loadMAT(slice_filename,parameters_filename):
    '''
    Created to convert .mat files with specific configuration for ECoG data of Newcastle Hospitals and create a dict.
    If you want to load other .mat file, use scipy.io. loadmat and create_DataObj
    '''
    mat = sio.loadmat(parameters_filename, struct_as_record=False, squeeze_me=True)
    parameters = mat['parameters']
    ch_labels = parameters.channels
    sample_rate = parameters.sr
    f = sio.loadmat(slice_filename, struct_as_record=False, squeeze_me=True)
    Data = f['Data']
    time_vec = Data.time_vec
    signal = Data.raw.T
    amp_unit = '$\mu V$'
    DataObj = create_DataObj(signal,sample_rate,amp_unit,ch_labels,time_vec,[])
    return DataObj
    
    
def loadSPK_waveclus(filename):
    '''
    load Spikes sorted by wave_clus.
    filename - Str with file .mat
    '''
    SpkObj = SpikeList()
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    clusters = mat['cluster_class'][:,0]
    times = mat['cluster_class'][:,1]/1000
    spikes = mat['spikes']
    features = mat['inspk']
    for idx,waveform in enumerate(spikes):
        tstamp = times[idx]
        clus = clusters[idx]
        feat= features[idx]
        spk = SpikeObj(waveform,tstamp,clus,feat)
        SpkObj.__addEvent__(spk)
    return SpkObj
    
    