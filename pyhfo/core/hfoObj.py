# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:03:56 2015

@author: anderson
"""

class hfoObj(object):
    def __repr__(self):
        return self.htype

    def __init__(self,channel,tstamp,tstamp_idx, waveform,start_idx,end_idx,info):
        self.htype = 'HFO'
        self.channel = channel              # channel  
        self.tstamp = tstamp                # Time stamp in sec. (center of event- max amp)
        self.tstamp_idx = tstamp_idx        # Time stamp in points. (center of event- max amp)
        self.waveform = waveform            # numpy array shape (sample_rate,2). 
                                            # first col - 1 second of raw wave
                                            # second col - 1 second of filtered wave
                                            # centered in tstamp (0.5 sec before and 0.5 after)
        self.start_idx = start_idx          # start index - when in waveform start HFO
        self.end_idx = end_idx              # end index - when in waveform end HFO
        self.info = info                    # info about the method of detection (cuttofs, order)
        