#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 03:48:40 2022

@author: cuilab
"""

import quantities as pq
import os
import glob
import numpy as np
import joblib
from MongoNeo.NPDInterface.BehaviorInterface import BehaviorMetaDataInterface
from MongoNeo.NPDInterface.NeuralDataInterface import BaseShare,FileConsolidation
import readTrodesExtractedDataFile3 as trodesReader
from MongoNeo.MongoNeoInterface import MongoNeoInterface

def SGEventMarkerReader(bhvfile):
    FileList = os.listdir(bhvfile)
    FileList.sort(key=lambda x : int(x[-5]), reverse=False)
    data = [trodesReader.readTrodesExtractedDataFile(os.path.join(bhvfile,i))['data']['state'] for i in FileList]
    time = [trodesReader.readTrodesExtractedDataFile(os.path.join(bhvfile,i))['data']['time'] for i in FileList]  
    
    timeq = np.unique(np.concatenate(time))
    marker = np.zeros((len(timeq),len(data)))
    marker[0,:] = np.array([i[0] for i in data])
    
    for i in range(1,marker.shape[0]):
        marker[i,:]=marker[i-1,:]
        for j in range(len(data)):
            if timeq[i] in time[j]:
                marker[i,j] = data[j][time[j]==timeq[i]]
    marker = marker[1::]
    markertime = (timeq-timeq[0])/30000*pq.s
    markertime = markertime[1::]
    marker = np.fliplr(marker).astype(int)
    decadeMarker = np.array([int(str(i)[1:-1].replace(' ', ''),2) for i in marker])
    return decadeMarker, markertime

class SGshare(BaseShare):
    def __init__(self, raw_dirname,
                 ComparedSeg='tdc_spike_train',
                 waveclus_dir_output=False):
        
        #NeuralDataName = os.path.join(raw_dirname,'NeuralData')
        BehaviorDataName = os.path.join(raw_dirname,'BehaviorData')
        #%% Get NEV & ML data path  
        db_file = glob.glob(raw_dirname+'/block.db')
        if len(db_file)>0:
            with open(db_file[0],"rb") as f:
                blockdata = joblib.load(f)            
            self.block_br = blockdata['block_br']
            self.mldata = BehaviorMetaDataInterface(BehaviorDataName,os.path.split(raw_dirname)[-1]).block
            return
        
        self.raw_dirname = raw_dirname
        # self.mldata = BehaviorMetaDataInterface(raw_dirname,os.path.split(raw_dirname)[-1]).block
        
        LFPRecordingParam = None
        LFPData = None
        LFPSampling_rate = None

        sampling_rate = 30000*pq.Hz
        
        nev_dir = os.path.join(BehaviorDataName, '*DIO*')
        nev_dir = glob.glob(nev_dir)[0]
        event_marker, event_time = SGEventMarkerReader(nev_dir)
        
        cp = list(np.random.rand(1024,2))
        cm = list(range(1024))
        RawRecordingParam = {}
        for m,p in zip(cm,cp):    
            RawRecordingParam[m]={}
            RawRecordingParam[m]['pos'] = p.squeeze()
            
        super().__init__(os.path.split(raw_dirname)[-1],
                         event_marker,
                         event_time,
                         RawRecordingParam,
                         sampling_rate,
                         LFPData,
                         LFPSampling_rate,
                         LFPRecordingParam,
                         raw_dirname,
                         ComparedSeg,
                         waveclus_dir_output)
        
    def _EMG_segment(self,EMG_path):
        pass
    
    def _Motion_segment(self,Motion_path):
        pass

dirname = '/home/cuilab/Desktop/hotdata/20220311_online_decoding.kilosort'
FileConsolidation(dirname,['DIO','bhv'])
SGData = SGshare(dirname)
mni = MongoNeoInterface(os.path.split(dirname)[-1],dbName="Hilbert_monkey_threads",
                        MongoAdress="mongodb://localhost:27017/",
                        username='admin', password='cuilab324')

mni.Sent2Mongo(SGData.block_br)
mni.Sent2Mongo(SGData.mldata)


