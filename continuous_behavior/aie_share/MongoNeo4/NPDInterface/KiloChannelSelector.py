#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:19:32 2022

@author: cuilab
"""
import numpy as np
from scipy import signal
from tqdm import tqdm

sec = 150

fp = np.memmap('/home/cuilab/Desktop/hotdata/20220311_online_decoding.kilosort/20220311_online_decoding.group0.dat', dtype='int16', mode='r')
fp = fp.reshape((int(fp.shape[0]/1024),1024))
fpSlice = np.array(fp[0:30000*sec,:])

wn=2*300/30000
b, a = signal.butter(8, wn, 'highpass')
filtedData = [signal.filtfilt(b, a, fpSlice[:,i]) for i in tqdm(range(fpSlice.shape[1]))]
del fpSlice
threshold = 6

SpikeList = []
for i in tqdm(filtedData):
    median = np.nanmedian(i)
    diff = np.abs(i - median)
    median_diff = np.median(diff)
    Zt = 0.6745 * diff / median_diff
    crossedPoint = np.where(Zt>threshold)[0]
    SpikeList.append([j for j in crossedPoint if Zt[j-1] < threshold])
    
SelectedSpikeList = [(chind, ch) for chind, ch in enumerate(SpikeList) if len(ch)>sec]
        
SimilarityMatrix = np.zeros((len(SelectedSpikeList),len(SelectedSpikeList)))
for ini, i in tqdm(enumerate(SelectedSpikeList)):
    for inj, j in enumerate(SelectedSpikeList):
        if len(j[1])/len(i[1])<0.5 or len(j[1])/len(i[1])>2:continue
        TimeBin1 = np.round(np.array(i[1])/30)
        TimeBin2 = np.round(np.array(j[1])/30)
        SimilarityMatrix[ini,inj] = len(set(TimeBin1)&set(TimeBin2))/len(set(TimeBin1))
        
CountMatrix = (SimilarityMatrix>0.7).sum(1)-1
CountMatrix = CountMatrix<20

SelectedResults = [i for ind, i in enumerate(SelectedSpikeList) if CountMatrix[ind]] 
SelectedResults = [i for i in SelectedResults if i[0]>500 and i[0]<900]
SelectedChannels = np.array([i[0] for i in SelectedResults])
fp = np.memmap('/home/cuilab/Desktop/hotdata/20220311_online_decoding.kilosort/20220311_online_decoding.group0.dat', dtype='int16', mode='r')
fp = fp.reshape((int(fp.shape[0]/1024),1024))

SampleInterval = 30000*sec
fpSlice = fp[0:SampleInterval,SelectedChannels]
for i in tqdm(range(30000*sec,fp.shape[0],30000*sec)):
    fpSlice = np.concatenate((fpSlice,fp[i:i+30000*sec,SelectedChannels])) if (fp.shape[0]-i)>SampleInterval else np.concatenate((fpSlice,fp[i::,SelectedChannels]))
fp = np.memmap('/home/cuilab/Desktop/hotdata/20220311_online_decoding.kilosort/Sliced.dat', dtype='int16', mode='write',shape=fpSlice.shape)
fp[:] = fpSlice
fp = np.memmap('/home/cuilab/Desktop/hotdata/20220311_online_decoding.kilosort/Sliced.dat', dtype='int16', mode='r')
