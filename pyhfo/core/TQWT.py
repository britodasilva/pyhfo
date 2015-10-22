# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:57:57 2015

@author: anderson
"""

import numpy as np
import math
from __future__ import division

def lps(x,N0):
    N = X.shape[1]
    Y = np.zeros(1,N0)
    
    if N0 <= N:
        k = range(N0/2-1)
        Y[k+1] = x[k+1]
        Y[N0/2 + 1] = x(N/2 + 1)
        k = 

def ComputeNow(N,Q,r,J = None, radix2 = True):
    if radix2:
        if np.log2(N) != np.round(np.log2(N)):
            raise Exception('N must be a power of 2 for radix-2 option for computing norm of wavelets')
    beta = 2/(Q+1)
    alpha = 1-beta/r
    if J is None:
        J = np.floor(np.log(beta*N/8)/np.log(1/alpha))
    w = dict()
    for j in np.arange(J):
        N0 = int(2*np.round(alpha**j * N/2))
        N1 = int(2*np.round(beta * alpha**(j-1) * N/2))
        if radix2:
            w[j] = np.zeros((1,1<<(N1-1).bit_length()))
        else:
            w[j] = np.zeros((1,N1))
    if radix2:
        w[J] = np.zeros((1,1<<(N0-1).bit_length()))
    else:
        w[J] = np.zeros((1,N0))
    now = np.zeros((1,J+1))
    wz = w
    for j in np.arange(J+1):
        w = wz
        M = w[j].shape[1]
        w[j][:] = 1/math.sqrt(M)
        Y = w[J]
        if radix2:
            M = 2*np.round(alpha**J * N/2)
            Y = 
    
    
    