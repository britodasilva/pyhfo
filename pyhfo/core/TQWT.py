# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:57:57 2015

@author: anderson
"""

from __future__ import division
import numpy as np

def lps(X,N0):
    N0 = int(N0)
    N = X.shape[0]
    Y = np.zeros((N0,))

    if N0 <= N:
        k = range(int(N0/2))
        Y[k] = X[k]
        Y[N0/2] = X[N/2]
        k = np.arange(1,int(N0/2))
        Y[N0-k] = X[N-k]

    elif N0> N:
        k = range(int(N0/2))
        Y[k] = X[k]
        k = np.arange(int(N/2),int(N0/2))
        Y[k] = 0
        Y[N0/2] = X[N/2]
        Y[N0-k] = 0
        k = np.arange(1,int(N/2))
        Y[N0-k] = X[N-k]
    return Y 
        
def sfb(V0,V1,N):
    N0 = int(V0.shape[0])
    N1 = int(V1.shape[0])
    N = int(N)
    S = int((N-N0)/2)
    P = int((N-N1)/2)
    T = int((N0+N1-N)/2 - 1   )
    v = np.arange(1,T+1)/(T+1)*np.pi

    trans = np.multiply( (1+np.cos(v)) , np.sqrt(2-np.cos(v))/2)

    Y0 = np.zeros(N)
    Y0[0] = V0[0]
    Y0[np.arange(P)] = V0[np.arange(P)]
    Y0[P+np.arange(T)] = np.multiply(V0[P+np.arange(T)],trans)
    Y0[P+T+np.arange(S)] = 0
    if N%2 == 0:
        Y0[N/2] = 0
    Y0[N-P-T-np.arange(S)-1] = 0
    Y0[N-P-np.arange(T)-1] = np.multiply(V0[N0-P-np.arange(T)-1],trans)
    Y0[N-np.arange(P)-1] = V0[N0-np.arange(P)-1]
    
    Y1 = np.zeros(N)
    Y1[0] = 0
    Y1[np.arange(P)] = 0
    Y1[P+np.arange(T)] = np.multiply(V1[np.arange(T)],trans[np.arange(T-1,-1,-1)])
    Y1[P+T+np.arange(S)] = V1[T+np.arange(S)]
    if N%2 == 0:
        Y1[N/2] = V1[N1/2]
    Y1[N-P-T-np.arange(S)-1] = V1[N1-T-np.arange(S)-1]
    Y1[N-P-np.arange(T)-1] = np.multiply(V1[N1-np.arange(T)-1],trans[np.arange(T-1,-1,-1)])
    Y1[N-np.arange(P)-1] = 0
                  
    return Y0+Y1

def ComputeNow(N,Q,r,J = None, radix2 = True):
    
    if radix2:
        if np.log2(N) != np.round(np.log2(N)):
            raise Exception('N must be a power of 2 for radix-2 option for computing norm of wavelets')
    beta = 2/(Q+1)
    alpha = 1-beta/r
    
    if J is None:
        J = int(np.floor(np.log(beta*N/8)/np.log(1/alpha)))
        print J
    w = dict()
    for j in np.arange(J):
        N0 = int(2*np.round(alpha**(j+1) * N/2))
        N1 = int(2*np.round(beta * alpha**j * N/2))
        
        if radix2:
            w[j] = np.zeros((1,1<<(N1-1).bit_length()))
        else:
            w[j] = np.zeros((1,N1))
    if radix2:
        w[J] = np.zeros((1,1<<(N0-1).bit_length()))
    else:
        w[J] = np.zeros((1,N0))
    now = np.zeros(J+1)
    wz = w
    for i in np.arange(J+1):
        w = wz
        M = w[i].shape[1]
        w[i][:] = 1/np.sqrt(M)
        Y = w[J][0]
        if radix2:
            M = 2*np.round(alpha**J * N/2)
            Y = lps(Y,M)
        for j in range(J-1,-1,-1):
            W = w[j][0]

            if radix2:
                N1 = int(2*np.round(beta * alpha**j * N/2))
                W = lps(W,N1)
            M = 2*np.round(alpha**j * N/2)
            Y = sfb(Y, W, M)
        now[i] = np.sqrt(np.sum(np.abs(Y)**2))
    return now
    
    
    