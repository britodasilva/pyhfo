# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:01:08 2015

@author: anderson
"""

class EventList(object):
    event = [] 
    def __addEvent__(self,obj):
        self.event.append(obj)
    def __removeEvent__(self,idx):
        del self.event[idx]
    def __repr__(self):
        return '%s events' % len(self.event)