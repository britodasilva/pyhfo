# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:01:08 2015

@author: anderson
"""
import numpy as np
from .IndexObj import IndexObj
from pyhfo.ui import plot_single_spk, plot_spk_cluster, adjust_spines
from IPython.html import widgets # Widget definitions
from IPython.display import display, clear_output # Used to display widgets in the notebook
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches


class EventList(object):
    event = [] 
    htype = 'list'
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
    
    def plot_event(self, ev = 0,figure_size = (5,5), dpi=600,**kwargs):
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
        **kwargs: matplotlib arguments
        """
        def f_button(clicked):
            idx = event.add(1)
            clear_output()
            if self.event[idx].htype == 'Spike':
                plot_single_spk(self.event[idx], figure_size=figure_size, dpi = dpi,**kwargs)
            plt.title('Event ' + str(idx))
        
        def b_button(clicked):
            idx = event.add(-1)
            clear_output()
            if self.event[idx].htype == 'Spike':
                plot_single_spk(self.event[idx], figure_size=figure_size, dpi = dpi,**kwargs)  
            plt.title('Event ' + str(idx))
        
        # Creating the figure 
        f = plt.figure(figsize=figure_size,dpi=dpi)
        # creating the axes
        ax = f.add_subplot(111)
        if self.event[ev].htype == 'Spike':
            plot_single_spk(self.event[ev], subplot = ax, figure_size=figure_size, dpi = dpi,**kwargs) 
        plt.title('Event ' + str(ev))
        #plt.close(fig)
        event = IndexObj(ev)
            
        buttonf = widgets.Button(description = ">>")
        buttonb = widgets.Button(description = "<<")
            
        buttonf.on_click(f_button)
        buttonb.on_click(b_button)
        vbox = widgets.Box()
        vbox.children = [buttonb,buttonf]        
        display(vbox)
        
        
    def plot_cluster(self,cluster=0,color='b', spines = [], plot_mean = True,figure_size=(5,5),dpi=600):
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
        htypes = self.__getlist__('htype')
        if 'HFO' in htypes:
            raise Exception('HFO htype not accepted')
            
        def f_button(clicked):
            idx = event.add(1)
            clear_output()
            plot_spk_cluster(self,idx,color=color, spines = spines, plot_mean = plot_mean, figure_size=figure_size, dpi = dpi)
            plt.title('Cluster ' + str(idx))
        
        def b_button(clicked):
            idx = event.add(-1)
            clear_output()
            plot_spk_cluster(self,idx,color=color, spines = spines, plot_mean = plot_mean, figure_size=figure_size, dpi = dpi) 
            plt.title('Cluster ' + str(idx))
        
        # Creating the figure 
        f = plt.figure(figsize=figure_size,dpi=dpi)
        # creating the axes
        ax = f.add_subplot(111)
        plot_spk_cluster(self,cluster,color=color,ax = ax, spines = spines, plot_mean = plot_mean)
        plt.title('Cluster ' + str(cluster))
        #plt.close(fig)
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
            
            
    def rastergram(self, ax = None, spines = ['left'],time_vec = None,ch_labels = None, exclude = [],figure_size=(15,5),dpi=600):
        """
        Plot rastergram 
        
        Parameters
        ----------
        ax: matplotlib axes 
            None (default) - create a new figure
            ax - axes of figure where figure should plot
        spines: str
            ['left', 'bottom'] (default) - plot figure with left and bottom spines only
        time_vec: vector or tupple
            Determine the x-axis limits. Need at least 2 values, if more (a vector), get the first and the last values.
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
            cluster  = self.__getlist__('channel')
        num_clus = int(np.max(cluster))+1
        
        label = []
        c_l = 0
        for clus in [x for x in range(num_clus) if x not in exclude]:
            if htype == 'Spike':
                if ch_labels == None:
                    label.append('Cluster ' + str(clus))
                else:
                    label.append(ch_labels[clus])
                objs = [x for x in self.event if x.cluster == clus]
            elif htype == 'HFO':
                if ch_labels == None:
                    label.append('Channel ' + str(clus))
                else:
                    label.append(ch_labels[clus])
                objs = [x for x in self.event if x.channel == clus]
                       
            for ev in objs:
                rect = patches.Rectangle((ev.tstamp,c_l),0.001,.8, lw=0.5) 
                ax.add_patch(rect)
            c_l += 1
        tstamp = self.__getlist__('tstamp')
        ax.set_ylim(0,c_l)
        if time_vec != None:
            ax.set_xlim(time_vec[0],time_vec[-1])
        else:
            ax.set_xlim(tstamp[0],tstamp[-1])
        plt.yticks(np.arange(c_l)+0.5,label, size=16)
        adjust_spines(ax, spines)