# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:43:48 2015

@author: anderson
core __init__
"""
import DataObj
import EventList
import SpikeObj
import hfoObj
from pre_processing import decimate, resample, merge, add_bad, remove_bad, create_avg, eegfilt