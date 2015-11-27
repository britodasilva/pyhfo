# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:33:52 2015

@author: anderson
"""
import numpy as np

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