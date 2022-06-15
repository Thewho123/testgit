#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 00:34:34 2021

@author: user
"""
from MongoNeo.BaseInterfaceTools import str2dtype
import pymongo
from MongoNeo.NetWorkFinder.NetworkModuleFP32Grid3Dim import CCHModule,CatchCCHMl
import quantities as pq
import numpy as np
from elephant.kernels import GaussianKernel
from MongoNeo.MongoReader import MongoReadModule
from tridesclous.iotools import ArrayCollection
import os
from tqdm import tqdm
import cupy
import random

class FrequenceAnalyzer(CCHModule):
    def MainFunc(self,**kargs):
        self.CCHConnectedMatrix(**kargs)
    def _FFT(self,CutOut):
        arrays = ArrayCollection(parent=None, dirname=os.path.join(self.Tempdir,self.collection))
        arrays.load_all()
        pbar = tqdm(list(arrays.keys()))
        BoolCutOut = np.zeros((self.MeanSurrogateCCH.shape[-1],))
        BoolCutOut[:] = True
        BoolCutOut[CutOut:-CutOut] = False
        TempArray = self.MeanSurrogateCCH.swapaxes(0,-1).swapaxes(1,-1)[BoolCutOut]
        FFTSurrogate = cupy.zeros(TempArray.shape)
        for i in pbar:
            CorrectedSurrogateCCH = arrays.get(i)-self.MeanSurrogateCCH
            FFTSurrogate[:] = CorrectedSurrogateCCH.swapaxes(0,-1).swapaxes(1,-1)[BoolCutOut]
            FFTSurrogateLog = cupy.asnumpy(cupy.log10(cupy.fft.fft(FFTSurrogate))).swapaxes(0,-1).swapaxes(0,1)[:,:,2:101]/2
            pbar.set_description("Processing GetSurrogateCCHFFT:")
        for i in pbar:
            arrays.detach_array(i)
            
            
if __name__ == '__main__':
    myclient = pymongo.MongoClient("mongodb://10.10.47.78:27017/", username='admin', password='cuilab324')
    mydb = myclient["Caesar_monkey"]
    #mkcollectiondir(CollectionList,Tempdir)
    CollectionList = mydb.list_collection_names()
    CollectionList = [i for i in CollectionList if i.find('202010')!=-1 and i.find('3T')==-1]
    #ListChunk = int(len(CollectionList)/3)
    #CollectionList = CollectionList[0:ListChunk]
    
    dtypeList = {}
    dtypeList['float32'] = np.float32
    dtypeList['float64'] = np.float64
    
    NetworkDict = {}
    for CollectionName in CollectionList:
    
        NetWorkFinder = CCHModule(collection = CollectionName,
                                  #SegName = 'TCR',
                                  SegName = 'FinalEnsemble_spike_train',
                                  Saverip="mongodb://10.10.47.78:27017/",
                                  db="Caesar_monkey")
        '''
        if '3T' in CollectionName:
            aligned_marker = 6
        else:
            aligned_marker = 5'''
            
        if '3T' in CollectionName:   
            continue
        
        aligned_marker = 5
        t_left = -800*pq.ms
        t_right = 0*pq.ms
        kernel = GaussianKernel(3.66*pq.ms)
        CCHcutoff = 200*pq.ms
        sampling_period = 1*pq.ms

        MongoReader = MongoReadModule(collection = CollectionName,
                                      SegName = 'FinalEnsemble_spike_train',
                                      #SegName = 'TCR',
                                      Saverip="mongodb://10.10.47.78:27017/",
                                      db='Caesar_monkey',
                                      username='yongxiang',
                                      password='cuilab322')
        
        #SelectedCondition = [{'DirectionIndex':0,'stage':3, 'objectInd':2,'DiscreateNum':4}]
        SelectedCondition = [{'UserVars':{'angularV':0}}]
        (FRSelectedTrial,STSelectedTrial) = CatchCCHMl(MongoReader,
                                                       t_left = t_left, 
                                                       t_right = t_right,
                                                       sampling_period = sampling_period,
                                                       aligned_marker = aligned_marker,
                                                       kernel = GaussianKernel(3.66*pq.ms),
                                                       CCHcutoff = 200*pq.ms,
                                                       TrialError=0,
                                                       #randnum = 500,
                                                       #replace = True,
                                                       SelectedCondition = SelectedCondition)
        CO_number = len(STSelectedTrial)
        
        #SelectedCondition = [{'UserVars':{'angularV': 0}}]
        
        UserKey = 'angularV'
        SpeedUserVar = [np.frombuffer(i[UserKey]['ArrayValue'],
                        dtype = str2dtype[i[UserKey]['ArrayDtype']]) for i in MongoReader.mldata['UserVars']]
        
        SelectedCondition = [{'UserVars':{'angularV':min(SpeedUserVar)}},{'UserVars':{'angularV':max(SpeedUserVar)[0]}}]
        (FRSelectedTrial,STSelectedTrial) = CatchCCHMl(MongoReader,
                                                       t_left = t_left, 
                                                       t_right = t_right,
                                                       sampling_period = sampling_period,
                                                       aligned_marker = aligned_marker,
                                                       kernel = GaussianKernel(3.66*pq.ms),
                                                       CCHcutoff = 200*pq.ms,
                                                       TrialError=0,
                                                       #randnum = 500,
                                                       #replace = True,
                                                       SelectedCondition = SelectedCondition)
        
        FRSelectedTrial = random.sample(FRSelectedTrial,CO_number)
        STSelectedTrial = random.sample(STSelectedTrial,CO_number)
        
        
      
    
        
        
        #%% Select Trial
        '''
        UserKey = 'angularV'
        FRSelectedTrial = []
        
        SpeedUserVar = [np.frombuffer(i['UserVars'][UserKey]['ArrayValue'],
                        dtype = str2dtype[i['UserVars'][UserKey]['ArrayDtype']]) for i in MongoReader.StatisticsML]
        SpeedUserVarUnique = [s for s in np.unique(SpeedUserVar) if s !=0]
        #SpeedList = [max(SpeedUserVarUnique),min(SpeedUserVarUnique)]
        SpeedList = [0]
        for speed in SpeedList:
            FRSelectedTrial = FRSelectedTrial+MongoReader.Get2DSelectedTrial(UserVars = {'angularV':speed})['SelectParameters']
        FRSelectedTrial = MongoReader.Get2DSelectedTrial(UserVars = {'angularV':speed})['SelectParameters']
        
        for speed in SpeedList:
            STSelectedTrial = STSelectedTrial+MongoReader.Get2DSelectedTrial(UserVars = {'angularV':speed})['SelectParameters']
        STSelectedTrial = MongoReader.Get2DSelectedTrial(UserVars = {'angularV':speed})['SelectParameters']'''
        #%% calculate CCH
        #STSelectedTrial = None
        #(FRSelectedTrial,STSelectedTrial) = (None,None)
        ConnectedMatrix = NetWorkFinder.CCHConnectedMatrix(sampling_period = 1*pq.ms, 
                                                           Tempdir = '/home/cuilab/Desktop/Caeser_TDC/CCHTemp',
                                                           t_left = t_left, 
                                                           t_right = t_right,
                                                           kernel = kernel,
                                                           aligned_marker = aligned_marker,
                                                           TrialError=0,
                                                           nbins=500,
                                                           tbin=0.001,
                                                           repNum = np.arange(0,10),
                                                           repBlock = 100,
                                                           st1sec = 0.8,
                                                           st2sec = 0.8,
                                                           CCHcutoff = 200*pq.ms,
                                                           Clustercutoff = 300,
                                                           FRStatisticsML = FRSelectedTrial,
                                                           STStatisticsML = STSelectedTrial,
                                                           permutation = 1000,
                                                           DeleteFile = True,
                                                           nickname='premovement_intercept')
        NetworkDict[CollectionName] = {}
        NetworkDict[CollectionName]['Matrix'] = np.frombuffer(ConnectedMatrix[0]['ConnectedMatrix']['bytes'],
                                                    dtypeList[ConnectedMatrix[0]['ConnectedMatrix']['dtype']]).\
                                                    reshape(ConnectedMatrix[0]['ConnectedMatrix']['shape'])
                                                    
        NetworkDict[CollectionName]['GroupIndexList'] = ConnectedMatrix[0]['GroupIndexList']
        '''
        save_matrix = NetworkDict[CollectionName]['Matrix']
        save_path = '/home/cuilab/Desktop/TempSave/CCHTemp/all_connectivity_matrix/'
        
        np.save(save_path + CollectionName, save_matrix)
        
        
        import pickle
        
        with open('/media/cuilab/Elements SE/Network_transfer/Caeser/-800-0_MO_MaxSpeed_new/NetworkDict.pickle', 'wb') as handle:
            pickle.dump(NetworkDict, handle)
        
        with open('/media/cuilab/Elements SE/NetworkDict.pickle','rb') as handle:
            unserialized_data = pickle.load(handle)
         
        '''