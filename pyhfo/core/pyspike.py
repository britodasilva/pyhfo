# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:05:54 2015

@author: anderson
"""

import numpy as np
from copy import copy
from DataObj import *
from EventList import *
from SpikeObj import *
import os
import matlab.engine
import scipy.io as sio
import shutil

def getSpike_from_DAT(folder,fname,ch,clus_folder,SPK):
    
    fh = open(folder+fname,'r')
    fh.seek(0)
    data = np.fromfile(fh, dtype=np.short, count=-1)
    fh.close()
    clear_clus_folder(clus_folder)
    name = fname[:-4] 
    data = np.double(data)
    sio.savemat(clus_folder+name+'.mat', {'data':data})
    f = open(clus_folder + 'Files.txt', 'w')
    f.write(name +'\n')
    f.close()
    eng = matlab.engine.start_matlab()
    eng.cd(clus_folder,nargout=0)
    eng.Get_spikes(nargout=0)
    eng.Do_clustering(nargout=0)
    move_figs(folder,clus_folder)
    eng.close('all', nargout=0)
    eng.quit()
    fname = clus_folder + 'times_' + name +'.mat'
    if os.path.isfile(fname): 
        SPK = loadSPK_waveclus(fname,SPK,ch)
    
    return SPK


def move_figs(folder,clus_folder):
    os.chdir(clus_folder)
    filelist = [ f for f in os.listdir(".") if f.endswith(".jpg") ]
    for f in filelist:
        shutil.move(clus_folder+f,folder+f)
    
def get_len(folder,fname):
    fh = open(folder+fname,'r')
    fh.seek(0)
    data = np.fromfile(fh, dtype=np.short, count=-1)
    fh.close()
    return data.shape[0]
    
def loadSPK_waveclus(filename,EventList,ch):
    '''
    load Spikes sorted by wave_clus.
    Parameters
    ----------
    filename: str
        Name of the spike (.mat) file 
    EventList: EventList
    
    '''
    
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    if mat['cluster_class'].size > 0:
        clusters = mat['cluster_class'][:,0]
        times = mat['cluster_class'][:,1]/1000
        spikes = mat['spikes']
        features = mat['inspk']
        labels = []
        for cl in range(int(max(clusters))+1):
            labels.append('Cluster '+str(cl))
        for idx,waveform in enumerate(spikes):
            tstamp = times[idx]
            clus = clusters[idx]
            feat= features[idx]
            time_edge = [-20,44]
            spk = SpikeObj(ch,waveform,tstamp,clus,feat,time_edge)
            EventList.__addEvent__(spk)
    return EventList

def clear_clus_folder(clus_folder):
    os.chdir(clus_folder)
    filelist = [ f for f in os.listdir(".") if f.endswith(".mat") ]
    for f in filelist:
        os.remove(f)
    filelist = [ f for f in os.listdir(".") if f.endswith(".mat500") ]
    for f in filelist:
        os.remove(f)
    filelist = [ f for f in os.listdir(".") if f.endswith(".mat400") ]
    for f in filelist:
        os.remove(f)
    filelist = [ f for f in os.listdir(".") if f.endswith(".lab") ]
    for f in filelist:
        os.remove(f)
    filelist = [ f for f in os.listdir(".") if f.endswith("01") ]
    for f in filelist:
        os.remove(f)
    filelist = [ f for f in os.listdir(".") if f.endswith("run") ]
    for f in filelist:
        os.remove(f)


    
def getFileList(folder):
    filelist = os.listdir(folder)
    label = [f for f in filelist if f.startswith('amp')]
    label.sort()
    return label
    
    
def open_file_DAT(folder,ports,nchans,srate,bsize=None,starttime = 0):
    if bsize == None:
        bsize = srate
    nch = sum(nchans)
    data = np.zeros([bsize,nch])
    count = 0
    labels = []
    
    for p in range(len(ports)):
        root =  'amp-'+ports[p]+'-'
        for ch in range(nchans[p]):
            x = str(ch)
            while len(x)<3:
                x = '0' + x
            fname = root + x + '.dat'
            labels.append(fname)
            fh = open(folder+fname,'r')
            fh.seek(np.round(starttime*srate)*2)
            data[:,count] = np.fromfile(fh, dtype=np.short, count=bsize)
            count +=1
            fh.close()
    data *= 0.195 # according the Intan, the output should be multiplied by 0.195 to be converted to micro-volts
    amp_unit = '$\mu V$'
    # Time vector   
    n_points  = data.shape[0]
    end_time  = n_points/srate
    time_vec  = np.linspace(0,end_time,n_points,endpoint=False)
    
    Data = DataObj(data,srate,amp_unit,labels,time_vec,[])
    return Data


def getDATduration(folder,port,ch):
    root =  'amp-'+port+'-'    
    x = str(ch)
    while len(x)<3:
        x = '0' + x
    fname = root + x + '.dat'
    fh = open(folder+fname,'r')
    data = np.fromfile(fh, dtype=np.short, count=-1)
    return data.shape[0]    
    
def pca2(data):

    npoint,ch = data.shape
    Mn = np.mean(data,0)
    for i in range(ch):
        data[:,i] = data[:,i] - Mn[i]
    
    C = np.cov(data.T)
    a1 = np.zeros(ch)
    a2 = np.zeros(ch)
    #art1 = np.zeros(npoint)
    
    #art2 = np.zeros(npoint)
    pcadata = np.zeros([npoint,ch])
    for i in range(ch):
        j = [x for x in range(ch) if x is not i]
        noti = data[:,j]
        Cnoti = C[np.ix_(j,j)]
        w,d = np.linalg.eig(Cnoti)
        k = np.argsort(w)
        d = d[k]
        v = np.identity(ch-1)*w
        v = v[k]
        v= v[:,-2:]
        pc = np.dot(noti,v)
        pc = np.append(pc,data[:,i][...,None],1)
        Cpc = np.cov(pc.T)

        a1[i] = Cpc[0,2]/Cpc[0,0]
        a2[i] = Cpc[1,2]/Cpc[1,1]
        #art1 += a1[i]*pc[:,0]
        #art2 += a2[i]*pc[:,1] 
        pcadata[:,i] = data[:,i] - a1[i]*pc[:,0] - a2[i]*pc[:,1] 
    #art1 /= ch
    #art2 /= ch
    #pca12 = np.append(art1[...,None],art2[...,None],1)
    
    return pcadata

def FindBigStuff(data,xsd =3,sd_method = 'Quian'):
    
    #s = np.std(data,0) * xsd
    
    #print s
    spikelist = np.array([0,0,0])[None,...]
    m,n = data.shape
    s = np.zeros(n)
    for i in range(n):
        
        x = data[:,i]
        if sd_method == 'Quian':
            s[i] = xsd * np.median(np.abs(x)) / 0.6745
        elif sd_method == 'STD':
            s[i] = np.std(x) * xsd
        taux = np.diff(np.where(abs(x)>s[i],1,0))
        times = np.nonzero(taux==1)[0]
        times2 = np.nonzero(taux==-1)[0]
        if len(times) !=0:
            if len(times)-1 == len(times2):
                times2 = np.append(times2,m)
            elif len(times) == len(times2)-1:
                times = np.append(0,times)
            chs = np.ones(times.shape)*i
            aux = np.append(chs[...,None],times[...,None],1)   
            aux = np.append(aux,times2[...,None],1)  
            spikelist = np.append(spikelist,aux,0)
    return np.delete(spikelist, (0), axis=0),s

def ReplaceBigStuff(data,biglist,replacearray,postpts = 10,prepts = 10):
    NoSpikesData = copy(data)
    for ch,atime,btime in biglist:
        if atime - prepts > 0:
            a = prepts
        else:
            a = atime-1
        if btime + postpts < data.shape[0]:
            b = postpts
        else:
            b = data.shape[0] - btime
        NoSpikesData[int(atime-a):int(btime+b),int(ch)] = replacearray[int(atime-a):int(btime+b),int(ch)]
    return NoSpikesData
    
    
def clearData(Data,ptspercut,postpts = 10,prepts = 10,xsd =3):
    data = Data.data
    m,n = data.shape
    if n>m:
        data = data.T
        m,n = data.shape
    last = m/ptspercut
    cleared = copy(data)
    for ci in range(last):
        if (ci+1)*ptspercut > m:
            stop = m
        else:
            stop = (ci+1)*ptspercut
        start = ci * ptspercut
        tdata = data[np.arange(start, stop),:,]
        
        
        pcadata = pca2(tdata)
        noiseEst = tdata - pcadata
        biglist,s = FindBigStuff(pcadata,xsd =3)
        replacearray = np.zeros(tdata.shape)
        NoSpikesData = ReplaceBigStuff(tdata,biglist,replacearray,postpts,prepts)
        
        pcadata = pca2(NoSpikesData)
        noiseEst = NoSpikesData - pcadata
        replacearray = noiseEst
        NoSpikesData = ReplaceBigStuff(tdata,biglist,replacearray,postpts,prepts)
        pcadata = pca2(NoSpikesData)
        cleared[np.arange(start, stop),:,] = tdata - pcadata
    Cleared = DataObj(cleared,Data.sample_rate,Data.amp_unit,Data.ch_labels,Data.time_vec,[])
    
    return Cleared
    
def GetSpike(Data,ptspercut=None,xsd=3,postpts = 10,prepts = 10, min_sep = None,sd_method = 'Quian'):
    time_edge = np.array([-prepts, postpts]) / float(Data.sample_rate)
    if min_sep is None:
        min_sep = float(prepts+postpts)/Data.sample_rate
    Spikes = EventList(Data.ch_labels,(Data.time_vec[0],Data.time_vec[-1]))  
    
    data = Data.data
    m,n = data.shape
    if n>m:
        data = data.T
        m,n = data.shape
    if ptspercut is None:
        ptspercut = Data.sample_rate
    last = m/ptspercut
    
    for ci in range(last):
        if (ci+1)*ptspercut > m:
            stop = m
        else:
            stop = (ci+1)*ptspercut
        start = ci * ptspercut
        tdata = data[np.arange(start, stop),:,]    
        biglist,ths = FindBigStuff(tdata,xsd =xsd,sd_method=sd_method)
        
        if biglist.shape[0]>0:
            ch_a = biglist[0,0]
        for ch,a,b in biglist:
            
            a += start
            b += start
            if ch == ch_a:
                tn = start
                tn /= float(Data.sample_rate)
                ch_a += 1
            if a - prepts > 0:
                pass
            else:
                
                continue
            if b + postpts < Data.data.shape[0]:
                pass
            else:
                continue
            
            aux = Data.data[int(a-prepts):int(b+postpts),int(ch)]
            
            #aux_idx = atime-a + np.argmax(abs(aux)) 
            #print np.nonzero(np.diff(np.where(abs(aux) > ths[int(ch)],1,0))==1)[0][0]
            #aux_idx = a-prepts + np.nonzero(abs(aux) > ths[int(ch)])[0][0]
            if len(np.nonzero(np.diff(np.where(abs(aux) > ths[int(ch)],1,0))==1)[0]) ==0:
                a -= prepts
                aux = Data.data[int(a-prepts):int(b+postpts),int(ch)]                
            aux_idx = a-prepts +  np.nonzero(np.diff(np.where(abs(aux) > ths[int(ch)],1,0))==1)[0][0] 
            
            waveform = Data.data[int(aux_idx-prepts):int(aux_idx+postpts),int(ch)]
            
            if waveform.shape[0] != postpts+prepts:
                continue
            tstamp = aux_idx/Data.sample_rate
            
            if tstamp - tn < min_sep:
                #print ch, tstamp,tn, tstamp-tn, min_sep
                continue
            tn = tstamp
            clus = 0
            feat = 0
            spk = SpikeObj(ch,waveform,tstamp,clus,feat,time_edge)
            Spikes.__addEvent__(spk)
    return Spikes
    
