#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:47:44 2021

@author: cuilab
"""

import numpy as np
from MongoNeo.MongoReader import MongoReadModule
import quantities as pq
from elephant.kernels import GaussianKernel
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from pyod.models.knn import KNN
from dPCA import dPCA
from sklearn.model_selection import KFold
from pyod.models.mad import MAD
import pymongo
import PSID
from tqdm import tqdm
import warnings
from MongoNeo.BaseInterfaceTools import GenerateQuery

warnings.filterwarnings('ignore')
#%% Cell Select
def CellSelect(FiringRate,MOTimeIndex,PeriMOTimeIndex):
    CellSelectIndex = []
    for i in range(FiringRate.shape[-1]):
        PeriActivity = FiringRate[:,PeriMOTimeIndex,i]
        MOActivity = FiringRate[:,MOTimeIndex,i]
        Result = ttest_ind(PeriActivity,MOActivity)
        if Result.pvalue<0.01:
            Diff = MOActivity-PeriActivity
            CrossZero = max(sum(Diff>0),sum(Diff<0))
            if CrossZero>(0.75*FiringRate.shape[0]):
                CellSelectIndex = CellSelectIndex+[i]
    return CellSelectIndex

#%% some tools
def FeatureExtraction(FiringRate,FeatureProjectionMatrix):
    FeatureMatrix = np.array([FeatureProjectionMatrix@i.transpose() for i in FiringRate]).swapaxes(1,2)
    return FeatureMatrix

#%% Decomposition method
def PCAFeatureSpace(FiringRate):
    pca = PCA()
    
    X_train = []
    
    for i in range(FiringRate.shape[-1]):
        X_train.append(FiringRate[:,:,i].flatten())
        
    pca.fit(np.array(X_train).transpose())
    return pca

def MeanFeatureSpace(FiringRate):
    return np.ones((FiringRate.shape[-1],))/FiringRate.shape[-1]

def dPCAFeatureSpace(**kargs):
    
    FiringRate = kargs['FiringRate']
    DiscreateNum = len(np.unique(kargs['stimulus']))
    
    if 'Desicion' not in kargs.keys():
    # number of neurons, time-points and stimuli
        N,T,S,n_samples = FiringRate.shape[-1],FiringRate.shape[1],DiscreateNum,FiringRate.shape[0]
        # build trial-by trial data
        trialR = np.zeros((n_samples,N,S,T))
        TrialIndex = np.arange(0,n_samples)
        for i in np.unique(kargs['stimulus']):
            BootstrapIndex = np.random.choice(TrialIndex[kargs['stimulus']==i],size = (n_samples,))
            trialR[:,:,int(i),:] = FiringRate[BootstrapIndex].swapaxes(1,2)
            
    else:
        DiscreateDesicionNum = len(np.unique(kargs['Desicion']))
        N,T,S,D,n_samples = FiringRate.shape[-1],FiringRate.shape[1],DiscreateNum,DiscreateDesicionNum,FiringRate.shape[0]
        # build trial-by trial data
        trialR = np.zeros((n_samples,N,S,D,T))
        TrialIndex = np.arange(0,n_samples)
        for i in np.unique(kargs['stimulus']):
            for j in np.unique(kargs['Desicion']):
                BootstrapIndex = np.random.choice(TrialIndex[kargs['stimulus']==i and kargs['Desicion']==j],size = (n_samples,))
                trialR[:,:,int(i),int(j),:] = FiringRate[BootstrapIndex].swapaxes(1,2)
            
    # trial-average data
    R = np.mean(trialR,0)
    
    # center data
    R -= np.mean(R.reshape((N,-1)),1)[:,None,None]
    
    dpca = dPCA.dPCA(labels='st',regularizer='auto')
    dpca.protect = ['t']
    Z = dpca.fit(R,trialR)
    return Z.D['t'][:,0]
#%% Classifier
def KNNTrainer(FiringRate):
    X_train = []
    
    for i in range(FiringRate.shape[-1]):
        X_train.append(FiringRate[:,:,i].flatten())
        
    clf = KNN(n_jobs = -1)
    clf.fit(np.array(X_train).transpose())
    return clf    
#%% Decoder design  
def DetectResultPos(DetectorResult,CulmulativeNum):
    PosList = []
    for i in range(DetectorResult.shape[0]):
        Pos = np.where(DetectorResult[i]==1)[0]
        if Pos.shape[0]!=0:
            
            for PosInd in range(Pos.shape[0]):
                if (PosInd+CulmulativeNum)>=Pos.shape[0]:
                    PosList = PosList+[-1]
                    break
                else:
                    if CulmulativeNum==0:
                        PosList = PosList+[Pos[0]]
                        break
                    else:
                        if sum(np.diff(Pos)[PosInd:PosInd+CulmulativeNum])==CulmulativeNum:
                            PosList = PosList+[Pos[PosInd+CulmulativeNum]]
                            break
                    
        else:
            PosList = PosList+[-1]
    return np.array(PosList)

def evaluation(DetectorPos):
    Evaluation = {}
    Evaluation['FalseNeg'] = sum(DetectorPos==-1)/DetectorPos.shape[0]
    Evaluation['Std'] = DetectorPos[DetectorPos!=-1].std()
    Evaluation['Mean'] = DetectorPos[DetectorPos!=-1].mean()
    return Evaluation

#%% Main module body
class MovementTimingFinder(MongoReadModule):
    
    # Preprocessing method 
    def DetectorFinder(self,**kargs):
        '''
        Method for data preprocessing in PD model fitting. 

        Parameters
        ----------
        **kargs : Dict
            Send Parameters for data preprocessing to find self.StatisticsML & self.PopulationFiringRate
        kargs containing the following parameters:
            'Statistics' : {'instantaneous_rate','time_histogram'}, spike statistics method
            't_left':{float*quantities_units}, left boundary of time window 
            't_right':{float*quantities_units}, right boundary of time window 
            'aligned_marker':float
            'TrialError':float
            'StatisticsML':{None,List}, Selected trial for analysis
            'cutoff':float, time bin cutoff number

        Returns
        -------
        None.

        '''
        #self.SpikeStatisticsReader(**kargs)
        #%% Preprocessing
        if 'FiringRate' not in kargs.keys():
            self.SpikeStatisticsReader(**kargs)
            if 'instantaneous_rate' in kargs['Statistics']:
                self.PopulationFiringRate = self.FiringRate
                
            if 'time_histogram' in kargs['Statistics']:
                self.PopulationFiringRate = self.SpikeBinList
        #%% Sent results to class property
        self.SetPopulationFiringRateKargs = kargs.copy()
        self.DetectorKargs = kargs.copy()
        if 'FiringRate' not in kargs.keys():
            self.DetectorKargs['FiringRate'] = self.PopulationFiringRate[:,kargs['cutoff']:-kargs['cutoff'],:]
        else:
            self.DetectorKargs['FiringRate'] = self.DetectorKargs['FiringRate'][:,kargs['cutoff']:-kargs['cutoff'],:]
        self.DetectorTrainer(**self.DetectorKargs)
        if 'cv' in kargs.keys():
            self.DetectorKargs['CVResults'] = self.CrossValidation(**self.DetectorKargs)
    #%% Traing function
    def DetectorTrainer(self,**kargs):
        #%% PCA MAD
        if kargs['Detecter']=='PCA':
            self.DetectorKargs['DetecterModel'] = PCAFeatureSpace(kargs['FiringRate']) 
        #%% KNN
        if kargs['Detecter']=='KNN':
            self.DetectorKargs['DetecterModel'] = KNNTrainer(kargs['FiringRate'])   
        #%% dPCA
        if kargs['Detecter']=='dPCA':        
            self.DetectorKargs['DetecterModel'] = dPCAFeatureSpace(**kargs)
        #%% Mean
        if kargs['Detecter']=='Mean':  
            self.DetectorKargs['DetecterModel'] = MeanFeatureSpace(kargs['FiringRate'])
        #%% PSID
        if kargs['Detecter']=='PSID':  
            self.DetectorKargs['DetecterModel'] = PSID.PSID(list(kargs['FiringRate']), 
                                                            kargs['BehaviorList'], 
                                                            nx=4, n1=3, i=4)
    #%% Predict function
    def DetectorPredictor(self,**kargs):
        #%% PCA MAD
        if kargs['Detecter']=='PCA':
            FeatureMatrix = np.array([self.DetectorKargs['DetecterModel'].transform(i) for i in kargs['FiringRate']])
            Projection = np.linalg.norm(FeatureMatrix,axis=2)  
            Detector = MAD(threshold=kargs['threshold'])
            Detector.fit(Projection.flatten()[:,np.newaxis]) 
            DetectorResult = Detector.predict(Projection.flatten()[:,np.newaxis]).reshape(Projection.shape)
            DetectorPos = DetectResultPos(DetectorResult,kargs['CulmulativeNum'])
            return DetectorPos
        #%% KNN
        if kargs['Detecter']=='KNN':
            if kargs['FiringRate'].shape[1]>15:
                cutoff = int(kargs['FiringRate'].shape[1]/4)
                FiringRateCutoff = kargs['FiringRate'][:,cutoff:-cutoff,:]
            X_test = []
            for i in range(FiringRateCutoff.shape[-1]):
                X_test.append(FiringRateCutoff[:,:,i].flatten())
            X_test = np.array(X_test).transpose()
            DetectorResult = self.DetectorKargs['DetecterModel'].predict(X_test).reshape(FiringRateCutoff.shape[0:2])   
            DetectorPos = DetectResultPos(DetectorResult,kargs['CulmulativeNum'])
            return DetectorPos
        #%% dPCA
        if kargs['Detecter']=='dPCA':
            Projection = np.array([i@self.DetectorKargs['DetecterModel'] for i in kargs['FiringRate']])
            Detector = MAD(threshold=kargs['threshold'])
            Detector.fit(Projection.flatten()[:,np.newaxis]) 
            DetectorResult = Detector.predict(Projection.flatten()[:,np.newaxis]).reshape(Projection.shape)
            DetectorPos = DetectResultPos(DetectorResult,kargs['CulmulativeNum'])
            return DetectorPos
        #%% Mean
        if kargs['Detecter']=='Mean':  
            Projection = np.array([i@self.DetectorKargs['DetecterModel'] for i in kargs['FiringRate']])
            Detector = MAD(threshold=kargs['threshold'])
            Detector.fit(Projection.flatten()[:,np.newaxis]) 
            DetectorResult = Detector.predict(Projection.flatten()[:,np.newaxis]).reshape(Projection.shape)
            DetectorPos = DetectResultPos(DetectorResult,kargs['CulmulativeNum'])
            return DetectorPos
        #%% PSID
        if kargs['Detecter']=='PSID':  
            Projection = np.array([self.DetectorKargs['DetecterModel'].predict(ti)[-1][:,-1] for ti in kargs['FiringRate']])[:,5:-5]
            Detector = MAD(threshold=kargs['threshold'])
            Detector.fit(Projection.flatten()[:,np.newaxis]) 
            DetectorResult = Detector.predict(Projection.flatten()[:,np.newaxis]).reshape(Projection.shape)
            DetectorPos = DetectResultPos(DetectorResult,kargs['CulmulativeNum'])
            return DetectorPos
    #%% training module
    def DetectorTraining(self,**kargs):
        ThreList = []
        StdList = []
        FalseNegIndex = 0
        if 'KNN' not in kargs['Detecter']:
            self.DetectorTrainer(**kargs)
            while len(StdList)==0:
                FalseNegIndex = FalseNegIndex+0.05
                for i in np.arange(0.1,5.1,0.1):
                    kargs['threshold'] = i
                    for j in range(1):
                        kargs['CulmulativeNum'] = j
                        DetectorPos = self.DetectorPredictor(**kargs)
                        Eva = evaluation(DetectorPos)
                        if (Eva['FalseNeg'] < FalseNegIndex) and (Eva['Mean']>5):
                            StdList.append(Eva['Std'])
                            ThreList.append((i,j))
                        
            ThreCul = [i for n,i in enumerate(ThreList) if StdList[n] == min(StdList)][0]
            
            for underthred in np.arange(0.1,ThreCul[0]+0.01,0.1):
                kargs['threshold'] = underthred
                for undercul in range(1):
                    kargs['CulmulativeNum'] = undercul
                    DetectorPos = CollectionDetector.DetectorPredictor(**kargs)
                    Eva = evaluation(DetectorPos)
                    if np.round(min(StdList)/Eva['Std'],2)>=0.95 and (Eva['Mean']>5):
                        break
                if np.round(min(StdList)/Eva['Std'],2)>=0.95 and (Eva['Mean']>5):
                    break
                    
            self.DetectorKargs['threshold'] = kargs['threshold']
            self.DetectorKargs['CulmulativeNum'] = ThreCul[1]
            
        else:
                       
            if kargs['FiringRate'].shape[1]>15:
                cutoff = int(kargs['FiringRate'].shape[1]/4)
                FiringRateCutoff = kargs['FiringRate'][:,cutoff:-cutoff,:]
            
            kargs['FiringRate'] = FiringRateCutoff
            self.DetectorTrainer(**kargs)

            X_test = []
            for i in range(FiringRateCutoff.shape[-1]):
                X_test.append(FiringRateCutoff[:,:,i].flatten())
            X_test = np.array(X_test).transpose()
            KNNDetectorResultTest = self.DetectorKargs['DetecterModel'].predict(X_test).\
                reshape(FiringRateCutoff.shape[0:2])
            
            while len(StdList)==0:
                FalseNegIndex = FalseNegIndex+0.1
                for j in range(2):
                    kargs['CulmulativeNum'] = j
                    DetectorPos = DetectResultPos(KNNDetectorResultTest,j)
                    Eva = evaluation(DetectorPos)
                    if (Eva['FalseNeg'] < FalseNegIndex) & (Eva['Mean']>5):
                        StdList.append(Eva['Std'])
                        ThreList.append(j)
            ThreCul = [i for n,i in enumerate(ThreList) if StdList[n] == min(StdList)][0]
            self.DetectorKargs['CulmulativeNum'] = ThreCul
    #%% CrossVal
    def CrossValidation(self,**kargs):
        SelectVar = {}
        for key in self.SetPopulationFiringRateKargs.keys():
            if 'FiringRate' in key:
                continue
            if 'stimulus' in key:
                continue
            if 'BehaviorList' in key:
                continue
            SelectVar[key] = self.SetPopulationFiringRateKargs[key]
        InsertDict = {}
        InsertDict['name'] = 'DetectorCV'
        GenerateQuery(InsertDict,SelectVar)
        GenerateQuery(InsertDict,self.StatisticsKargs)
        #self.testID = InsertDict
        if 'StatisticsML' in InsertDict.keys(): del InsertDict['StatisticsML']
        CVResults = [i for i in self.mycol.find(InsertDict)]
        if len(CVResults)!=0: 
            self.DetectorKargs[kargs['Detecter']+'Eva']=CVResults[0][kargs['Detecter']+'Eva']
            return
        
        kf = KFold(n_splits=kargs['cv'])
        self.DetectorKargs[kargs['Detecter']+'Eva'] = []
        self.DetectorKargs['Detecter'] = kargs['Detecter']
        FiringRate = self.DetectorKargs['FiringRate'].copy()
        if 'stimulus' in self.DetectorKargs.keys():
            stimulus = self.DetectorKargs['stimulus'].copy()
        if 'BehaviorList' in self.DetectorKargs.keys():
            BehaviorList = self.DetectorKargs['BehaviorList'].copy()
            
        for train, test in kf.split(range(FiringRate.shape[0])):
            self.DetectorKargs['FiringRate'] = FiringRate[train]
            
            if 'stimulus' in self.DetectorKargs.keys():
                self.DetectorKargs['stimulus'] = stimulus[train]
            if 'BehaviorList' in self.DetectorKargs.keys():
                self.DetectorKargs['BehaviorList'] = [BehaviorList[index] for index in train]      
                
            self.DetectorTraining(**self.DetectorKargs)
            
            self.DetectorKargs['FiringRate'] = FiringRate[test]
            
            DetectorPos = self.DetectorPredictor(**self.DetectorKargs)
            Eva = evaluation(DetectorPos)
            #Eva['threshold'] = self.DetectorKargs['threshold']
            self.DetectorKargs[kargs['Detecter']+'Eva'].append(Eva)  

        InsertDict[kargs['Detecter']+'Eva'] = self.DetectorKargs[kargs['Detecter']+'Eva']
        self.mycol.insert_one(InsertDict.copy())
#%% Main
if __name__ == '__main__':
    myclient = pymongo.MongoClient("mongodb://10.10.44.85:27017/", username='admin', password='cuilab324')
    mydb = myclient["Caesar_monkey"]
    #mkcollectiondir(CollectionList,Tempdir)
    CollectionList = mydb.list_collection_names()
    CollectionDetectorResults = {}
    CollectionDuration = {}
    CollectionDetectorCellSelectResults = {}
    DetectorResult = {}
    DetectorList = ['Mean','dPCA','PCA','KNN','PSID']
    for i in DetectorList:
        DetectorResult[i+'Eva'] = []
    for i in tqdm(CollectionList):
        
        if '3T' in i:
            continue
        
        CollectionDetector = MovementTimingFinder(collection = i,
                                                  SegName = 'FinalEnsemble_spike_train',
                                                  Saverip='mongodb://10.10.44.85:27017/',
                                                  db='Caesar_monkey', 
                                                  username='admin', 
                                                  password='cuilab324')
        
        
        for detector in DetectorList:
            if 'PSID' in detector:
                instantaneous_rate = CollectionDetector.SpikeFiringRate(Statistics = 'instantaneous_rate',
                                                                        sampling_period = 50*pq.ms, 
                                                                        t_left = -1000*pq.ms, 
                                                                        t_right = 1000*pq.ms,
                                                                        kernel = GaussianKernel(50*pq.ms),
                                                                        aligned_marker = 5,
                                                                        TrialError=0,
                                                                        StatisticsML = None)
                
                ml_data = CollectionDetector.Get2DStageObjectTrial(stage=3,objectInd=2)
                behavior_all = np.zeros((instantaneous_rate.shape[0],
                                         instantaneous_rate.shape[1]-10,3))
                Mid = int(behavior_all.shape[1]/2)
                for trii in range(behavior_all.shape[0]):
                    behavior_all[trii,Mid-3:Mid+3,0] = ml_data[trii][0]
                    behavior_all[trii,Mid-3:Mid+3,1] = ml_data[trii][1]
                    behavior_all[trii,Mid-3:Mid+3,2] = 16
                BehaviorList = list(behavior_all)
                CollectionDetector.DetectorFinder(FiringRate = instantaneous_rate,
                                                  BehaviorList = BehaviorList,
                                                  cv = 5,
                                                  Detecter = detector,
                                                  cutoff = 5)
                DetectorResult[detector+'Eva'] = DetectorResult[detector+'Eva']+\
                                            CollectionDetector.DetectorKargs[detector+'Eva']
                continue
                
            if 'dPCA' not in detector:
                CollectionDetector.DetectorFinder(Statistics = 'instantaneous_rate',
                                                  t_left = -1000*pq.ms, 
                                                  t_right = 1000*pq.ms,
                                                  kernel = GaussianKernel(50*pq.ms),
                                                  aligned_marker = 5,
                                                  TrialError=0,
                                                  sampling_period = 50*pq.ms,
                                                  StatisticsML = None,
                                                  cv = 5,
                                                  Detecter = detector,
                                                  cutoff=5)
                
                DetectorResult[detector+'Eva'] = DetectorResult[detector+'Eva']+\
                                            CollectionDetector.DetectorKargs[detector+'Eva']
                
                continue
            
            if 'dPCA' in detector:
                stimulus = []
                FiringRate = []
                
                instantaneous_rate = CollectionDetector.SpikeFiringRate(Statistics = 'instantaneous_rate',
                                                                        sampling_period = 50*pq.ms, 
                                                                        t_left = -1000*pq.ms, 
                                                                        t_right = 1000*pq.ms,
                                                                        kernel = GaussianKernel(50*pq.ms),
                                                                        aligned_marker = 5,
                                                                        TrialError=0,
                                                                        StatisticsML = None)
                
                for j in range(8):
                    SelectedTrial = CollectionDetector.Get2DSelectedTrial(DirectionIndex = j,stage = 3,objectInd = 2,
                                                                          DiscreateNum = 8)['SpikeStatisticSelect']
                    FiringRate.append(SelectedTrial)
                    stimulus.append(np.ones((SelectedTrial.shape[0],))*j)
                
                
                stimulus = np.concatenate(stimulus)
                FiringRate = np.concatenate(FiringRate)
                RandIndex = np.random.choice(range(stimulus.shape[0]),stimulus.shape[0],replace=False)
                stimulus = stimulus[RandIndex]
                FiringRate = FiringRate[RandIndex]
                
                CollectionDetector.DetectorFinder(FiringRate = FiringRate,
                                                  stimulus = stimulus,
                                                  cv = 5,
                                                  Detecter = detector,
                                                  cutoff = 5)
                DetectorResult[detector+'Eva'] = DetectorResult[detector+'Eva']+\
                                            CollectionDetector.DetectorKargs[detector+'Eva']



