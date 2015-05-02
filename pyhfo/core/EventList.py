# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:01:08 2015

@author: anderson
"""
import numpy as np

class EventList(object):
    event = [] 
    def __addEvent__(self,obj):
        self.event.append(obj)
    def __removeEvent__(self,idx):
        del self.event[idx]
    def __repr__(self):
        return '%s events' % len(self.event)
    def __getlist__(self,attr):
        '''
        return a list of atrribute
        '''
        attribute = np.array([])
        for ev in self.event:
            attribute = np.append(attribute,vars(ev)[attr])
        return attribute