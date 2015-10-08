# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:28:56 2015

@author: anderson
"""

import scipy.signal as sig
import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt

class HFOcoupling(object):
    def __init__(self,hfoObj):
        #signal = sig.detrend(hfoObj.waveform[hfoObj.start_idx:hfoObj.end_idx,0]) # detrending
        signal = sig.detrend(hfoObj.waveform[hfoObj.start_idx:hfoObj.end_idx,0])
        fs = hfoObj.sample_rate
        signal = sig.detrend(HFO.event[501].waveform[3*fs/4:5*fs/4,0])
        PhaseFreqVector= np.arange(1,31,1)
        AmpFreqVector= np.arange(30,990,5)
        PhaseFreq_BandWidth=1
        AmpFreq_BandWidth=10
        Comodulogram=np.zeros((PhaseFreqVector.shape[0],AmpFreqVector.shape[0]))
        nbin=18
        position=np.zeros(nbin)
        winsize = 2*np.pi/nbin
        for j in range(nbin):
            position[j] = -np.pi+j*winsize;
        PHASES = np.zeros((PhaseFreqVector.shape[0],signal.shape[0]))
        for idx,Pf1 in enumerate(PhaseFreqVector):
            print Pf1,
            Pf2 = Pf1 + PhaseFreq_BandWidth
            if signal.shape[0] > 18*np.fix(HFO.event[1].sample_rate/Pf1):
                b = sig.firwin(3*np.fix(HFO.event[1].sample_rate/Pf1),[Pf1,Pf2],pass_zero=False,window=('kaiser',0.5),nyq=HFO.event[1].sample_rate/2)
            else:
                b = sig.firwin(signal.shape[0]/6,[Pf1,Pf2],pass_zero=False,window=('kaiser',0.5),nyq=HFO.event[1].sample_rate/2)
            PhaseFreq = sig.filtfilt(b,np.array([1]),signal)
            Phase=np.angle(sig.hilbert(PhaseFreq))
            PHASES[idx,:]=Phase;
        print    
        for idx1,Af1 in enumerate(AmpFreqVector):
            print Af1,
            Af2 = Af1 + AmpFreq_BandWidth
            if signal.shape[0] > 18*np.fix(HFO.event[1].sample_rate/Af1):
                b = sig.firwin(3*np.fix(HFO.event[1].sample_rate/Af1),[Af1,Af2],pass_zero=False,window=('kaiser',0.5),nyq=HFO.event[1].sample_rate/2)
            else:
                b = sig.firwin(np.fix(signal.shape[0]/6),[Af1,Af2],pass_zero=False,window=('kaiser',0.5),nyq=HFO.event[1].sample_rate/2)
            AmpFreq = sig.filtfilt(b,np.array([1]),signal)
            Amp=np.abs(sig.hilbert(AmpFreq))
            for idx2,Pf1 in enumerate(PhaseFreqVector):
                Phase = PHASES[idx2]
                MeanAmp = np.zeros(nbin)
                for j in range(nbin):
                    bol1 = Phase < position[j]+winsize
                    bol2 = Phase >= position[j]
                    I = np.nonzero(bol1 & bol2)[0]
                    MeanAmp[j]=np.mean(Amp[I])
                #MI=(np.log(nbin)-(-np.sum((MeanAmp/np.sum(MeanAmp))*np.log((MeanAmp/np.sum(MeanAmp))))))/np.log(nbin)
                MI =np.log(nbin)-(stat.entropy(MeanAmp)/np.log(nbin))
                Comodulogram[idx2,idx1]=MI;
        plt.contourf(PhaseFreqVector+PhaseFreq_BandWidth/2,AmpFreqVector+AmpFreq_BandWidth/2,Comodulogram.T,100)