# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:03:56 2015

@author: anderson
"""
from __future__ import division
from pyhfo.ui import plot_single_hfo
from .HFOSpectrum import HFOSpectrum
import numpy as np
from book_reader import BookImporter
import matplotlib.pyplot as plt
import os

class hfoObj(object):
    def __repr__(self):
        return str(self.tstamp)

    def __init__(self,channel,tstamp,tstamp_idx, waveform,start_idx,end_idx,ths_value,sample_rate,cutoff,info):
        self.htype = 'HFO'
        self.channel = channel              # channel  
        self.tstamp = tstamp                # Time stamp in sec. (center of event- max amp)
        self.tstamp_idx = tstamp_idx        # Time stamp in points. (center of event- max amp)
        self.waveform = waveform            # numpy array shape (sample_rate,2). 
                                            # first col - 1 second of raw wave
                                            # second col - 1 second of filtered wave
                                            # centered in tstamp (0.5 sec before and 0.5 after)
        self.start_idx = start_idx          # start index - when in waveform start HFO
        self.start_sec = tstamp - (tstamp_idx-start_idx)/sample_rate  
        self.end_idx = end_idx              # end index - when in waveform end HFO
        self.end_sec = tstamp + (end_idx-tstamp_idx)/sample_rate
        self.ths_value = ths_value          # ths_value - value of choosen threshold
        self.sample_rate = sample_rate      # sample rate of recording
        self.cutoff = cutoff                # cutoff frequencies
        self.info = info                    # info about the method of detection (cuttofs, order)
        self.duration = self.end_sec - self.start_sec  # calculate the event duration
        self.spectrum = HFOSpectrum(self,cutoff) # spectrum object
        self.peak_amp = np.max(np.abs(waveform[:,1])) # get the peak amplitude value
        
        
    def plot(self,envelope = True, figure_size = (15,10),dpi=600):        
        plot_single_hfo(self, envelope = envelope, figure_size = figure_size,dpi=dpi)
        
        
    def __set_cluster__(self,cluster):
        self.cluster = cluster
        
    def MP(self):
        def creating_set_file(N,sr):
            f = open( 'MP_temp.set', 'w' )
            f.write('# OBLIGATORY PARAMETERS\n')
            f.write('nameOfDataFile          ' + name + '\n' )
            f.write('nameOfOutputDirectory   ' + './\n' )
            f.write('writingMode             ' + 'CREATE\n' )
            f.write('samplingFrequency       ' + repr(sr) + '\n' )
            f.write('numberOfChannels        ' + '1\n' )
            f.write('selectedChannels        ' + '1\n' )
            f.write('numberOfSamplesInEpoch  ' + repr(N) + '\n' )
            f.write('selectedEpochs          ' + '1\n' )
            f.write('typeOfDictionary        ' + 'OCTAVE_FIXED\n' )
            f.write('energyError             ' + '0.01 100.0\n' )
            f.write('randomSeed              ' + 'auto\n' )
            f.write('reinitDictionary        ' + 'NO_REINIT_AT_ALL\n' )
            f.write('maximalNumberOfIterations ' + '10\n' )
            f.write('energyPercent           ' + '99.\n' )
            f.write('MP                      ' + 'SMP\n' )
            f.write('scaleToPeriodFactor     ' + '1.0\n' )
            f.write('pointsPerMicrovolt      ' + '1000.\n' )
            f.write('# ADDITIONAL PARAMETERS\n' )
            f.write('normType                ' + 'L2\n' )
            f.write('diracInDictionary       ' + 'YES\n' )
            f.write('gaussInDictionary       ' + 'YES\n' )
            f.write('sinCosInDictionary      ' + 'YES\n' )
            f.write('gaborInDictionary       ' + 'YES\n' )
            f.write('progressBar             ' + 'OFF\n' )
            f.close()
        
        signal =  self.waveform[self.tstamp_idx-(self.sample_rate/4):self.tstamp_idx+(self.sample_rate/4),0]
        f_name = 'MP_temp'
        if os.path.isfile('MP_temp_smp.b'):
            os.system('rm MP_temp_smp.b')
        a = np.array(signal,'float32')
        name = f_name+'.dat'
        f = open(name,'wb')
        a.tofile(f)
        f.close()
        creating_set_file(signal.shape[0],float(self.sample_rate))
        
        os.system('mp5 -f MP_temp.set')
        
        book = BookImporter(f_name+'_smp.b')
        t      = np.linspace(0,len(signal)-1,len(signal))/book.fs
        Eatoms = 0
        reconstruction = np.zeros(len(t))
        #count = 0
        print self.ths_value
        for i,booknumber in enumerate(book.atoms):
            
            for atom in book.atoms[booknumber]:
                if atom['type'] !=13:
                    continue
                frequency = atom['params']['f']*book.fs/2
                amplitude = atom['params']['amplitude']
                phase 	  = atom['params']['phase']
                position  = atom['params']['t']/book.fs
                width     = atom['params']['scale']/book.fs
                #print frequency,width,position
                
                if frequency > 60 and frequency < 600 and position > self.start_idx/self.sample_rate and position < self.end_idx/self.sample_rate and width < .3:
                    print amplitude
                    reconstruction += amplitude*np.exp(-np.pi*((t-position)/width)**2)*np.cos(2*np.pi*frequency*(t-position)+phase)
                Eatoms += atom['params']['modulus'] 
        plt.subplot(311)
        plt.plot(t,signal); 
        plt.title('simulated signal from demo.dat')
        plt.subplot(312)
        plt.plot(t,book.signals[1][0]); 
        plt.title('signal stored in demo\_smp.b')
        # this is signal read from the *.b file, should be the same as the original
        plt.subplot(313)
        plt.plot(t,reconstruction,'r');
        plt.title('signal reconstructed from MP decomposition')
        plt.show()
        return book

   