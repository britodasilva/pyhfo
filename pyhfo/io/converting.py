# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:31:34 2015

@author: anderson
"""
import pyhfo as hfo
import os
import re


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    
    
def converting_intan_directory(main_directory,save_file):
    '''
    Get main folders, with slice folders inside, save HDF5 in save_file
    '''
    print bcolors.BOLD + main_directory + bcolors.ENDC
    # get slice folders name
    folders = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    for f in folders:
        print bcolors.OKGREEN + f + bcolors.ENDC
        # get the files
        files = os.listdir(main_directory + '/' + f)
        base_files = [s for s in files]
        for base in base_files:
            print bcolors.OKBLUE + 'File ' + base[-10:-4] + bcolors.ENDC
            dataset_name =  f + '/' + base[0:base.find('_')] + '/' + base[-10:-4]     
            RHD_name = main_directory + "/" + f + "/" + base
            # read files     
            Data = hfo.io.loadRDH(RHD_name)
            # save files            
            hfo.save_dataset(Data,save_file,dataset_name)
        