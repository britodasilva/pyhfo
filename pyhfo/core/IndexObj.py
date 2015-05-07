# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:40:58 2015

@author: anderson
"""

class IndexObj(object):
    def __init__(self,start_ind):
        self.ind = start_ind
    def add(self,num):
        self.ind += num
        return self.ind