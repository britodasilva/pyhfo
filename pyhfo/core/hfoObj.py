# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:03:56 2015

@author: anderson
"""

class hfoObj(object):
    def __repr__(self):
        return self.htype

    def __init__(self,htype,ch,tstamp,other):
        self.htype = htype
        self.tstamp = tstamp
        self.ch = ch
        self.other = other     