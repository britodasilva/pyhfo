# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:01:08 2015

@author: anderson
"""
import numpy as np
from .IndexObj import IndexObj
from pyhfo.ui import plot_single_spk, plot_spk_cluster, adjust_spines,plot_single_hfo, plot_mean_hfo
#from pyhfo.io.o_functions import save_dataset
from IPython.html import widgets # Widget definitions
from IPython.display import display, clear_output # Used to display widgets in the notebook
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from sklearn.cluster import KMeans

class EventList(object):
    def __init__(self,ch_labels,time_edge,file_name = None, dataset_name = None):
        self.htype = 'list'
        self.event = []
        self.ch_labels = ch_labels
        self.time_edge = time_edge
        if file_name != None:
            self.filename = file_name
        if dataset_name != None:
            self.datasetname = dataset_name
    def __addEvent__(self,obj):
        self.event.append(obj)
    def __removeEvent__(self,idx):
        if type(idx) == int:
            del self.event[idx]
        else:
            indexes = sorted(list(idx), reverse=True)
            for index in indexes:
                del self.event[index]
        
    def delete_cluster(self,cluster):
        clu = self.__getlist__('cluster')
        to_remove = np.nonzero(clu == cluster)[0]
        self.__removeEvent__(to_remove)
    def __repr__(self):
        return '%s events' % len(self.event)
    def __getlist__(self,attr):
        '''
        return a list of atrribute
        '''
        attribute = np.array([])

        for ev in self.event:
            if hasattr(ev, attr):
                attribute = np.append(attribute,vars(ev)[attr])
            elif attr == 'entropy':
                attribute = [x.entropy for x in self.__getlist__('spectrum')]
            elif attr == 'peak_freq':
                attribute = [x.peak_freq for x in self.__getlist__('spectrum')]
            elif attr == 'power_index':
                attribute = [x.power_index for x in self.__getlist__('spectrum')]
            elif attr == 'peak_win_power':
                attribute = [x.peak_win_power for x in self.__getlist__('spectrum')]
            else:
                raise Exception('Attribute not found')
            
        
        return attribute
    
    def plot_event(self, ev = 0,figure_size = (5,5), dpi=600,xlim=[-1,1],saveplot = None, cutoff = False ,**kwargs):
        """
        Plot events
        
        Parameters
        ----------
        ev: int
            event number
        subplot: matplotlib axes 
            None (default) - create a new figure
            ax - axes of figure where figure should plot
        spines: str
            ['left', 'bottom'] (default) - plot figure with left and bottom spines only
        figure_size: tuple
            (5,5) (default) - Size of figure, tuple of integers with width, height in inches 
        dpi: int
            600 - DPI resolution
        cutoff: boolean
            True (default) -  Fill the cuttof frequency in spectogram
        **kwargs: matplotlib arguments
        """
        def ploting(self,idx):
            clear_output()
            if self.event[idx].htype == 'Spike':
                plot_single_spk(self.event[idx], figure_size=figure_size, dpi = dpi,saveplot=saveplot,**kwargs)
            
            if self.event[idx].htype == 'HFO':
                if cutoff:
                    cut = self.event[ev].cutoff
                else:
                    cut = None
                plot_single_hfo(self.event[idx], figure_size = figure_size,dpi=dpi,xlim=xlim,cutoff = cut,saveplot=saveplot)
            plt.suptitle('Event ' + str(idx))
            
        def f_button(clicked):
            idx = event.add(1)
            ploting(self,idx)

        
        def b_button(clicked):
            idx = event.add(-1)
            ploting(self,idx)


        ploting(self,ev)

        #plt.close(fig)
        event = IndexObj(ev)
            
        buttonf = widgets.Button(description = ">>")
        buttonb = widgets.Button(description = "<<")
            
        buttonf.on_click(f_button)
        buttonb.on_click(b_button)
        vbox = widgets.Box()
        vbox.children = [buttonb,buttonf]        
        display(vbox)
        
        
    def plot_cluster(self,cluster=0,color='blue', spines = [], plot_mean = True,xlim =[-1,1], figure_size=(10,10),dpi=600,saveplot = None):
        """
        Plot spike cluster. If event list contains HFO raise error. 
        
        Parameters
        ----------
        cluster: int
            Number of the cluster
        color: str
            Color of plot
        spines: str
            ['left', 'bottom'] (default) - plot figure with left and bottom spines only
        plot_mean: boolean
            True (default) - plot mean line
        figure_size: tuple
            (5,5) (default) - Size of figure, tuple of integers with width, height in inches 
        dpi: int
            600 - DPI resolution
        """
#        htypes = self.__getlist__('htype')
#        if 'HFO' in htypes:
#            raise Exception('HFO htype not accepted')

        def ploting(self,idx):
            clear_output()
            if self.event[0].htype == 'Spike':
                plot_spk_cluster(self,idx,color=color, spines = spines, plot_mean = plot_mean, figure_size=figure_size, dpi = dpi)
                
            if self.event[0].htype == 'HFO':
                evlist = [self.event[x] for x in range(len(self.event)) if self.event[x].cluster==idx]
                plot_mean_hfo(evlist, color = color,  xlim =xlim, figure_size=figure_size,dpi=dpi,saveplot=saveplot)
            plt.suptitle('Cluster ' + str(idx))
            
        def f_button(clicked):
            idx = event.add(1)
            ploting(self,idx)
            
                
        
        def b_button(clicked):
            idx = event.add(-1)
            ploting(self,idx)
            
        
        if self.event[0].htype == 'HFO':
            if not hasattr(self.event[0],'cluster'):
                self.HFO_clustering()
            
        
        ploting(self,cluster)

        event = IndexObj(cluster)
            
        buttonf = widgets.Button(description = ">>")
        buttonb = widgets.Button(description = "<<")
            
        buttonf.on_click(f_button)
        buttonb.on_click(b_button)
        vbox = widgets.Box()
        vbox.children = [buttonb,buttonf]        
        display(vbox)
        
    def plot_all_spk_clusters(self,plot_mean = True,figure_size=(10,10),dpi=600):
        """
        Plot all spike cluster. If event list contains HFO raise error. 
        
        Parameters
        ----------
        plot_mean: boolean
            True (default) - plot mean line
        figure_size: tuple
            (5,5) (default) - Size of figure, tuple of integers with width, height in inches 
        dpi: int
            600 - DPI resolution
        """
        if len(self.event) == 0:
            raise Exception('No events to plot')
        htypes = self.__getlist__('htype')
        if 'HFO' in htypes:
            raise Exception('HFO htype not accepted')
        cluster = self.__getlist__('cluster')
        num_clus = int(np.max(cluster))+1
        ncols = int(math.ceil(math.sqrt(num_clus)))
        nrows = int(math.floor(math.sqrt(num_clus)))
        fig,sb = plt.subplots(nrows,ncols,sharey=True,figsize=figure_size,dpi=dpi)
        c = 0
        l = 0
        for clus in range(num_clus):
            if c == ncols:
                c = 0
                l += 1
            plot_spk_cluster(self, clus, ax = sb[l,c])
            sb[l,c].set_title('Cluster ' + str(clus))
            c +=1
            
            
    def rastergram(self, ax = None, spines = ['left','bottom'],time_edge = None, exclude = [],figure_size=(15,5),dpi=600, line = True):
        """
        Plot rastergram 
        
        Parameters
        ----------
        ax: matplotlib axes 
            None (default) - create a new figure
            ax - axes of figure where figure should plot
        spines: str
            ['left', 'bottom'] (default) - plot figure with left and bottom spines only
        time_edge: tupple
            Determine the x-axis limits. 
        exclude: list 
            Channels/Cluster to exclude from plot
        figure_size: tuple
            (5,5) (default) - Size of figure, tuple of integers with width, height in inches 
        dpi: int
            600 - DPI resolution
        """
       
        htype = self.event[0].htype
        if ax == None:
             # Creating the figure 
            f = plt.figure(figsize=figure_size,dpi=dpi)
            # creating the axes
            ax = f.add_subplot(111)
        if htype == 'Spike':
            cluster  = self.__getlist__('cluster')
        elif htype == 'HFO':
            cluster  = range(len(self.ch_labels))
        if time_edge == None:
            time_edge = self.time_edge
                
        num_clus = int(np.max(cluster))+1
        
        label = []
        c_l = 0
        for clus in [x for x in range(num_clus) if x not in exclude]:
            label.append(self.ch_labels[clus])
            if htype == 'Spike':
                objs = [x for x in self.event if x.cluster == clus]
               
            elif htype == 'HFO':
                objs = [x for x in self.event if x.channel == clus]
            for ev in objs:
                rect = patches.Rectangle((ev.tstamp,c_l),0.001,.8, lw=0.5) 
                ax.add_patch(rect)
                
            if line:
                plt.hlines(c_l+.5,time_edge[0],time_edge[1],colors='c')
            c_l += 1
        ax.set_ylim(0,c_l)
        ax.set_xlim(time_edge[0],time_edge[-1])
        plt.yticks(np.arange(c_l)+0.5,label, size=16)
        adjust_spines(ax, spines)
        
        
#    def save(self,filename = None,datasetname = None ):
#        if filename == None:
#            if hasattr(self,'filename'):
#                filename = self.filename
#                
#            else:
#                raise Exception('No file name')
#                
#        if datasetname == None:
#            if hasattr(self,'datasetname'):
#                datasetname = self.datasetname
#            else:
#                raise Exception('No dataset name')        
#     
#        save_dataset(self,filename,datasetname)
        
        
    def HFO_clustering(self,n_clusters = 2, attr = ['entropy']):
        #clustering 
        at = np.array([])
        for attribute in attr:
            new_at = np.array(self.__getlist__(attribute))
            at = np.append(at,new_at)
        at = np.reshape(at,(len(attr),len(new_at)))
        y= KMeans(n_clusters=n_clusters).fit(at.T)
        for ix, hfo in enumerate(self.event):
            hfo.__set_cluster__(y.labels_[ix])


    def reset_cluster(self):
        for hfo in self.event:
            hfo.__set_cluster__(0)