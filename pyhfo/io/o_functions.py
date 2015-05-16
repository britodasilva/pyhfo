# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:10:01 2015

@author: anderson
"""
import h5py


def save_dataset(Obj,file_name,obj_name):
    '''
    save Obj in a dataset in a specific file_name
       
    Parameters
    ----------
    obj: DataObj, list
        DataObj to save
        list object to save
    file_name: str 
        Name of the HDF5 (.h5) file 
    obj_name: str
        if obj is DataOnj, Name of dataset; 
        if obj is a list, Name of group;
    '''
    # open or creating file 
    h5 = h5py.File(file_name,'a')
    # deleting previous dataset
    if obj_name in h5:
        del h5[obj_name]
    # cheking kind of Obj
    if Obj.htype == 'Data':
        dataset  = h5.create_dataset(obj_name,data=Obj.data)
        dataset.attrs.create('htype',Obj.htype)
        dataset.attrs.create('SampleRate[Hz]',Obj.sample_rate)
        dataset.attrs.create('amp_unit',Obj.amp_unit)
        dataset.attrs.create('Bad_channels',Obj.bad_channels)
        dataset.attrs.create('Channel_Labels', Obj.ch_labels)
        dataset.attrs.create('Time_vec_edge',[Obj.time_vec[0],Obj.time_vec[-1]])
    elif Obj.htype == 'list':
        group = h5.create_group(obj_name)
        group.attrs.create('htype',Obj.htype)
        group.attrs.create('time_edge',[Obj.time_edge[0],Obj.time_edge[-1]])
        group.attrs.create('ch_labels', Obj.ch_labels[:])
        for idx, ev in enumerate(Obj.event):
            name = ev.htype + '_' + str(idx)
            if ev.htype == 'Spike':
                dataset  = group.create_dataset(name,data=ev.waveform)
                dataset.attrs.create('htype',ev.htype)
                dataset.attrs.create('tstamp',ev.tstamp)
                dataset.attrs.create('cluster',ev.cluster)
                dataset.attrs.create('features',ev.features)
            elif ev.htype == 'HFO':
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
            
    h5.close()