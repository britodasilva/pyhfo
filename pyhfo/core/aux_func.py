# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:33:52 2015

@author: anderson
"""
import numpy as np
import time
from pyhfo.core import EventList

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
    
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %.4f seconds' % (time.time() - self.tstart)
        
        
def merge_lists(EvList1,EvList2):
    ch_labels1 = EvList1.ch_labels
    ch_labels2 = EvList2.ch_labels
    if set(ch_labels1) != set(ch_labels2):
        raise 'Merge should be from same channels list'
    time_edge1 = EvList1.time_edge
    time_edge2 = EvList2.time_edge
    if time_edge1[1]<time_edge2[0]:
        new_time = time_edge1[0],time_edge2[1]
    else:
        new_time = time_edge2[0],time_edge1[1]
        
    NewEvList = EventList(ch_labels1,new_time)
    for ev in EvList1:
        NewEvList.__addEvent__(ev)
    for ev in EvList2:
        NewEvList.__addEvent__(ev)
    return NewEvList