#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:12:28 2020

@author: cuilab
"""



from neo import Segment
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import  partial
from affinewarp import SpikeData

def ListCompare(Cell,CellList):
    CellSet = set(np.where(Cell.bin_spikes(int(Cell.tmax)).squeeze())[0])
    CellListSet = [set(np.where(i.bin_spikes(int(i.tmax)).squeeze())[0]) for i in CellList]
    interCellListSet = [len(CellSet&i)/len(CellSet) for i in CellListSet]
    return interCellListSet

def TwoSorterCompare(kilosortSeg,tdcSeg):
    mlPool = Pool(20)
    tdcSpikeTimeList = [SpikeData(trials = np.zeros(np.array(i).shape[0]), 
                               spiketimes = np.array(i)*1000, 
                               neurons=np.zeros(np.array(i).shape[0]), 
                               tmin = 0, 
                               tmax = np.ceil(max(np.array(i)*1000))) for i in tqdm(tdcSeg)]
        
    kiloSpikeTimeList = [SpikeData(trials = np.zeros(np.array(i).shape[0]), 
                           spiketimes = np.array(i)*1000, 
                           neurons=np.zeros(np.array(i).shape[0]), 
                           tmin = 0, 
                           tmax = np.ceil(max(np.array(i)*1000))) for i in tqdm(kilosortSeg)]
    
    SpiketrainList = mlPool.map(partial(ListCompare,
                                        CellList=kiloSpikeTimeList),tqdm(tdcSpikeTimeList))
    mlPool.close()
    mlPool.join()
    SpiketrainList = np.array(SpiketrainList)
    return SpiketrainList

def GetEnsemble(ComparedSeg,SegnameList,blockdata):
    SegDict = {}
    
    for Seg in blockdata.segments:
        if Seg.name in SegnameList:
            SegDict[Seg.name]=Seg.spiketrains
                
    CompareDict = {}
    #for i in list(SegList.keys()):

    CompareDict[ComparedSeg] = []
    for j in SegDict.keys():
        if ComparedSeg==j:
            continue
        CompareDict[ComparedSeg].append(TwoSorterCompare(SegDict[ComparedSeg],SegDict[j]))
        
    if len(CompareDict[ComparedSeg])>4:
        SegNames = [i.name for i in blockdata.segments]
        MaxDict = {}
        for i in CompareDict.keys():
            MaxList = np.array([np.max(j,0) for j in CompareDict[i]])
            MaxList[MaxList>0.5] = 1
            MaxList[MaxList<0.5] = 0
            MaxList = MaxList.sum(0)
            MaxDict[i] = MaxList
        
        ComparedSegIndex = [n for n in range(len(SegNames)) if SegNames[n]==ComparedSeg][0]
        Seg = Segment(name = 'Ensemble_spike_train')
        for n,i in enumerate(blockdata.segments[ComparedSegIndex].spiketrains):
            if MaxDict[ComparedSeg][n]==0:
                continue
            if (MaxDict[ComparedSeg][n]<=2) and (i.description['snr']<5):
                continue
            spk_des = {'SimilarityLevel':MaxDict[ComparedSeg][n],
                       'SegID':n}
            i.description.update(spk_des)
            Seg.spiketrains.append(i)
            
        print('Cluster num:',len(Seg.spiketrains))
        blockdata.segments.append(Seg)
    else:
        print("not enough Seg")

    
    
    
    
    
    
    
    
    
    
    
    
    
    