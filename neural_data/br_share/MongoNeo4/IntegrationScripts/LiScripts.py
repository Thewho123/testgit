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
from MongoNeo.NPDInterface.BehaviorInterface import BehaviorMetaDataInterface,BaseShare,FileConsolidation
import h5py
from MongoNeo.MongoNeoInterface import MongoNeoInterface

class IMECshare(BaseShare):
    def __init__(self, raw_dirname,
                 ComparedSeg='tdc_spike_train',
                 waveclus_dir_output=False):
        NeuralDataName = os.path.join(raw_dirname,'NeuralData')
        BehaviorDataName = os.path.join(raw_dirname,'BehaviorData')
        #%% Get NEV & ML data path  
        db_file = glob.glob(raw_dirname+'/block.db')
        if len(db_file)>0:
            with open(db_file[0],"rb") as f:
                blockdata = joblib.load(f)            
            self.block_br = blockdata['block_br']
            self.mldata = BehaviorMetaDataInterface(raw_dirname,os.path.split(raw_dirname)[-1],mlpath=os.path.join(BehaviorDataName,'mldata.pkl')).block
            return
        
        self.raw_dirname = raw_dirname
        # self.mldata = BehaviorMetaDataInterface(raw_dirname,os.path.split(raw_dirname)[-1]).block
        
        NeuralFileList = os.listdir(NeuralDataName)

        try:
            LFPFileName = [i for i in NeuralFileList if '.lf.bin' in i][0]
            LFPMetaFileName = [i for i in NeuralFileList if '.lf.meta' in i][0]
            LFPData = np.loadtxt(os.path.join(NeuralDataName,LFPFileName))
            MetaData = {}
            with open(os.path.join(NeuralDataName,LFPMetaFileName)) as f:
                for i in f.readlines():
                    key,value = tuple(i.split('='))
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    MetaData[key] = value
            LFPRecordingParam = MetaData
            LFPSampling_rate = LFPRecordingParam['imSampRate']*pq.Hz
            LFPData = LFPData*pq.uV
        except:
            LFPRecordingParam = None
            LFPData = None
            LFPSampling_rate = None
            print('There does not exsits LFP file!')
        
        APMetaFileName = [i for i in NeuralFileList if '.ap.meta' in i][0]
        MetaData = {}
        with open(os.path.join(NeuralDataName,APMetaFileName)) as f:
            for i in f.readlines():
                key,value = tuple(i.split('='))
                try:
                    value = float(value)
                except ValueError:
                    pass
                MetaData[key] = value

        sampling_rate = MetaData['imSampRate']*pq.Hz
        
        nev_dir = os.path.join(NeuralDataName, 'events.hdf5')
        blk_event_marker = np.array(h5py.File(nev_dir,'r')['events'])
        event_time = blk_event_marker[:,0]/sampling_rate.magnitude*pq.sec
        event_marker = blk_event_marker[:,1]
        
        cp = list(np.load(os.path.join(NeuralDataName, 'channel_positions.npy')))
        cm = list(np.load(os.path.join(NeuralDataName, 'channel_map.npy')))
        RawRecordingParam = {}
        for m,p in zip(cm,cp):    
            RawRecordingParam[m[0]]={}
            RawRecordingParam[m[0]]['pos'] = p.squeeze()
            
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
                         waveclus_dir_output,
                         mlpath=os.path.join(BehaviorDataName,'mldata.pkl'))
        
        self.block_br.description.update(MetaData)
        
    def _EMG_segment(self,EMG_path):
        pass
    
    def _Motion_segment(self,Motion_path):
        pass

dirname = '/home/cuilab/Desktop/hotdata/raw_data_BMILAB/Lilab/M31_20201218_g0_imec0_cleaned'
FileConsolidation(dirname,['mldata.pkl'])
LiData = IMECshare(dirname)
LiData.save_block()