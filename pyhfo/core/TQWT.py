# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:57:57 2015

@author: anderson
"""

from __future__ import division
import numpy as np
from copy import deepcopy

def lps(X,N0):
    N0 = int(N0)
    N = int(X.shape[0])
    Y = np.zeros((N0,),dtype=complex)

    if N0 <= N:
        k = range(int(N0/2))
        Y[k] = X[k]
        Y[int(N0/2)] = X[int(N/2)]
        k = np.arange(1,int(N0/2))
        Y[N0-k] = X[N-k]

    elif N0> N:
        k = range(int(N0/2))
        Y[k] = X[k]
        k = np.arange(int(N/2),int(N0/2))
        Y[k] = 0
        Y[int(N0/2)] = X[int(N/2)]
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

    Y0 = np.zeros(N,dtype=complex)
    Y0[0] = V0[0]
    
    Y0[np.arange(1,P+1)] = V0[np.arange(1,P+1)]
    Y0[P+np.arange(1,T+1)] = np.multiply(V0[P+np.arange(1,T+1)],trans)
    Y0[P+T+np.arange(1,S+1)] = 0
    if N%2 == 0:
        Y0[int(N/2)] = 0
    Y0[N-P-T-np.arange(1,S+1)] = 0
    Y0[N-P-np.arange(1,T+1)] = np.multiply(V0[N0-P-np.arange(1,T+1)],trans)
    Y0[N-np.arange(1,P+1)] = V0[N0-np.arange(1,P+1)]
    
    
    Y1 = np.zeros(N,dtype=complex)
    Y1[0] = 0
    Y1[np.arange(1,P+1)] = 0
    Y1[P+np.arange(1,T+1)] = np.multiply(V1[np.arange(1,T+1)],trans[np.arange(T-1,-1,-1)])
    Y1[P+T+np.arange(1,S+1)] = V1[T+np.arange(1,S+1)]
    
    if N%2 == 0:
        Y1[int(N/2)] = V1[int(N1/2)]
    Y1[N-P-T-np.arange(1,S+1)] = V1[N1-T-np.arange(1,S+1)]
    Y1[N-P-np.arange(1,T+1)] = np.multiply(V1[N1-np.arange(1,T+1)],trans[np.arange(T-1,-1,-1)])
    Y1[N-np.arange(1,P+1)] = 0
                
    return Y0+Y1

def afb(X,N0,N1):
    N = int(X.shape[0])
    N0 = int(N0)
    N1 = int(N1)
    P = int((N-N1)/2)
    T = int((N0+N1-N)/2 - 1   )
    S = int((N-N0)/2)
    
    v = np.arange(1,T+1)/(T+1)*np.pi
    trans = np.multiply( (1+np.cos(v)) , np.sqrt(2-np.cos(v))/2)

    V0 = np.zeros(N0,dtype=complex)
    V0[0] = X[0]
    
    V0[np.arange(1,P+1)] = X[np.arange(1,P+1)]
    V0[P+np.arange(1,T+1)] = np.multiply(X[P+np.arange(1,T+1)],trans)
    V0[int(N0/2)] = 0
    V0[N0-P-np.arange(1,T+1)] = np.multiply(X[N-P-np.arange(1,T+1)],trans)
    V0[N0-np.arange(1,P+1)] = X[N-np.arange(1,P+1)]

    
    V1 = np.zeros(N1,dtype=complex)
    V1[0] = 0
    V1[np.arange(1,T+1)] = np.multiply(X[P+np.arange(1,T+1)],trans[np.arange(T-1,-1,-1)])
    V1[T+np.arange(1,S+1)] = X[P+T+np.arange(1,S+1)]
    
    if N%2 == 0:
        V1[int(N1/2)] = X[int(N/2)]
    
    V1[N1-T-np.arange(1,S+1)] = X[N-P-T-np.arange(1,S+1)]
    V1[N1-np.arange(1,T+1)] = np.multiply(X[N-P-np.arange(1,T+1)],trans[np.arange(T-1,-1,-1)])
    return V0,V1

def uDFTinv(X):
    N = X.shape[0]
    x = np.sqrt(N) * np.fft.ifft(X)
    return x

def uDFT(x):
    N = x.shape[0]
    X = np.fft.fft(x) / np.sqrt(N)
    return X

def ComputeNow(N,Q,r,J = None, radix2 = True):
    
    if radix2:
        if np.log2(N) != np.round(np.log2(N)):
            raise Exception('N must be a power of 2 for radix-2 option for computing norm of wavelets')
    beta = 2/(Q+1)
    alpha = 1-beta/r
   
    if J is None:
        J = int(np.floor(np.log(beta*N/8)/np.log(1/alpha)))
        #print 'J is equal ' + str(J)
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
    wz = w.copy()
    
    for i in np.arange(J+1):
        w = deepcopy(wz)
        
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
    return now,J
            
def tqwt_radix2(x,Q,r,J):
    beta = 2/(Q+1)
    alpha = 1-beta/r
    L = x.shape[0]
    N = 1<<(L-1).bit_length()
    Jmax = int(np.floor(np.log(beta*N/8)/np.log(1/alpha)))
    if J > Jmax:
        raise Exception('J higher than Jmax')
    X = np.fft.fft(x,N)/np.sqrt(N)
    
    w = dict()
    for j in np.arange(J):
        N0 = int(2*np.round(alpha**(j+1) * N/2))
        N1 = int(2*np.round(beta * alpha**j * N/2))
        X,W = afb(X,N0,N1)
        W = lps(W,1<<(N1-1).bit_length())
        w[j] = uDFTinv(W)
    X  = lps(X,1<<(N0-1).bit_length())
    w[J] = uDFTinv(X)
    return w

def itqwt_radix2(w,Q,r,L):
    beta = 2/(Q+1)
    alpha = 1-beta/r
    
    J = len(w.keys()) - 1
    N = 1<<(L-1).bit_length()
   
    Y = uDFT(w[J])
    M = int(2*np.round(alpha**J * N/2))
    Y = lps(Y,M)
    for j in range(J-1,-1,-1):
        W = uDFT(w[j])
        N1 = int(2*np.round(beta * alpha**j * N/2)) 
        W = lps(W,N1)
        M = int(2*np.round(alpha**j * N/2))
        Y = sfb(Y,W,M)
    y = uDFTinv(Y)
    y = y[range(L)]
    return y

def soft(x,T):
    y = np.fmax(np.abs(x)-T,0)
    y = np.multiply(np.divide(y,y+T),x)
    return y
    
def dualQd(x,Q1,r1,J1,Q2,r2,J2,lam1,lam2,mu,Nit):
    L = x.shape[0]
    N = 1<<(L-1).bit_length()
    if L<N:
        x = np.pad(x,(0,N-L),'constant')
    w1 = tqwt_radix2(x,Q1,r1,J1)
    w2 = tqwt_radix2(x,Q2,r2,J2)
    d1 = tqwt_radix2(np.zeros(x.shape[0]),Q1,r1,J1)
    d2 = tqwt_radix2(np.zeros(x.shape[0]),Q2,r2,J2)
    
    T1 = lam1/(2*mu)
    T2 = lam2/(2*mu)
    
    u1 = dict()
    u2 = dict()
    costfn = np.zeros(Nit)
    
    N = x.shape[0]
    
    for k in range(Nit):
        for j in range(J1):
            u1[j] = soft(w1[j] + d1[j], T1[j])- d1[j]
        
        for j in range(J2):
            u2[j] = soft(w2[j] + d2[j], T2[j])- d2[j]
            
        c = x - itqwt_radix2(u1,Q1,r1,N) - itqwt_radix2(u2,Q2,r2,N)
        c /= (mu+2) 
        
        d1 = tqwt_radix2(c,Q1,r1,J1)
        d2 = tqwt_radix2(c,Q2,r2,J2)
        
        costfn[k] = 0
        for j in range(J1):
            w1[j] = d1[j] + u1[j]
            costfn[k] = costfn[k] + lam1[j]*np.sum(np.abs(w1[j]))
            
        for j in range(J2):
            w2[j] = d2[j] + u2[j]
            costfn[k] = costfn[k] + lam2[j]*np.sum(np.abs(w2[j]))
            
    x1 = itqwt_radix2(w1,Q1,r1,L)
    x2 = itqwt_radix2(w2,Q2,r2,L)
    return x1,x2,w1,w2,costfn
    
def dualQ(x,Q1,r1,J1,Q2,r2,J2,lam1,lam2,mu,Nit):
    L = x.shape[0]
    N = 1<<(L-1).bit_length()
    if L<N:
        x = np.pad(x,(0,N-L),'constant')
    w1 = tqwt_radix2(x,Q1,r1,J1)
    w2 = tqwt_radix2(x,Q2,r2,J2)
    d1 = tqwt_radix2(np.zeros(x.shape[0]),Q1,r1,J1)
    d2 = tqwt_radix2(np.zeros(x.shape[0]),Q2,r2,J2)
    
    T1 = lam1/(2*mu)
    T2 = lam2/(2*mu)
    
    u1 = dict()
    u2 = dict()
    costfn = np.zeros(Nit)
    N = x.shape[0]
    
    for k in range(Nit):
        for j in range(J1):
            u1[j] = soft(w1[j] + d1[j], T1[j])- d1[j]
        
        for j in range(J2):
            u2[j] = soft(w2[j] + d2[j], T2[j])- d2[j]
            
        c = x - itqwt_radix2(u1,Q1,r1,N) - itqwt_radix2(u2,Q2,r2,N)
        c *= .5 
        
        d1 = tqwt_radix2(c,Q1,r1,J1)
        d2 = tqwt_radix2(c,Q2,r2,J2)
        costfn[k] = 0
        for j in range(J1):
            w1[j] = d1[j] + u1[j]
            costfn[k] = costfn[k] + lam1[j]*np.sum(np.abs(w1[j]))
            
        for j in range(J2):
            w2[j] = d2[j] + u2[j]
            costfn[k] = costfn[k] + lam2[j]*np.sum(np.abs(w2[j]))
            
       
        
            
    x1 = itqwt_radix2(w1,Q1,r1,L)
    x2 = itqwt_radix2(w2,Q2,r2,L)
    return x1,x2,w1,w2,costfn
    
def tqwt_bp(x,Q,r,J,lam,mu,Nit):
    L = x.shape[0]
    N = 1<<(L-1).bit_length()
    if L<N:
        x = np.pad(x,(0,N-L),'constant')
    w = tqwt_radix2(x,Q,r,J)
    d = tqwt_radix2(np.zeros(x.shape[0]),Q,r,J)
    
    T = lam/(2*mu)
    
    u = dict()
    costfn = np.zeros(Nit)
    N = x.shape[0]
    
    for k in range(Nit):
        for j in range(J):
            u[j] = soft(w[j] + d[j], T[j])- d[j]
                
        d = tqwt_radix2(x - itqwt_radix2(u,Q,r,N),Q,r,J)
        costfn[k] = 0
        for j in range(J):
            w[j] = d[j] + u[j]
            costfn[k] = costfn[k] + lam[j]*np.sum(np.abs(w[j]))
            
    y = itqwt_radix2(w, Q, r, N)
    return y, costfn
    
    
def tqwt_bpd(x,Q,r,J,lam,mu,Nit):
    L = x.shape[0]
    N = 1<<(L-1).bit_length()
    if L<N:
        x = np.pad(x,(0,N-L),'constant')
    w = tqwt_radix2(x,Q,r,J)
    d = tqwt_radix2(np.zeros(x.shape[0]),Q,r,J)
    
    T = lam/(2*mu)
    
    u = dict()
    costfn = np.zeros(Nit)
    N = x.shape[0]
    C = 1/(mu+1)
    for k in range(Nit):
        for j in range(J):
            u[j] = soft(w[j] + d[j], T[j])- d[j]
                
        d = tqwt_radix2(C*x - C*itqwt_radix2(u,Q,r,N),Q,r,J)
        costfn[k] = 0
        for j in range(J):
            w[j] = d[j] + u[j]
            costfn[k] = costfn[k] + lam[j]*np.sum(np.abs(w[j]))
            
    y = itqwt_radix2(w, Q, r, N)
    return y, costfn