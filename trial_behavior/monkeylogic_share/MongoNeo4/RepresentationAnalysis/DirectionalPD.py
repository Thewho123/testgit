#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:00:06 2021

@author: cuilab
"""


import numpy as np
from ..MongoReader import MongoReadModule
from sklearn import linear_model
#from joblib import Parallel,delayed
from tqdm import tqdm
import quantities as pq
#from elephant.kernels import GaussianKernel
import pymongo
from ..BaseInterfaceTools import SearchAndUpload

def BootstrapPDModel(FiringRate,Variable,bootstrap):
    
    PDResults = {}

    BootstrapSaver = {}
    BootstrapSaver['R2'] = []
    BootstrapSaver['BaseLine'] = []
    BootstrapSaver['ModulationDepth'] = []
    BootstrapSaver['PD'] = []
    
    
    for _ in range(bootstrap):
        PDModel = linear_model.LinearRegression()
        BootstrapIndex = np.random.randint(0,Variable.shape[0],(Variable.shape[0],))
        PDModel.fit(Variable[BootstrapIndex],FiringRate[BootstrapIndex])
        BootstrapSaver['R2'].append(PDModel.score(Variable[BootstrapIndex],FiringRate[BootstrapIndex]))
        BootstrapSaver['BaseLine'].append(PDModel.intercept_)
        BootstrapSaver['ModulationDepth'].append(np.linalg.norm(PDModel.coef_))
        BootstrapSaver['PD'].append(np.math.atan2(PDModel.coef_[1],PDModel.coef_[0]))
    
    PDResults['R2'] = {}
    PDResults['R2']['mean'] = np.mean(BootstrapSaver['R2'])
    PDResults['R2']['std'] = np.std(BootstrapSaver['R2'])
    
    PDResults['BaseLine'] = {}
    PDResults['BaseLine']['mean'] = np.mean(BootstrapSaver['BaseLine'])
    PDResults['BaseLine']['std'] = np.std(BootstrapSaver['BaseLine'])
    
    PDResults['ModulationDepth'] = {}
    PDResults['ModulationDepth']['mean'] = np.mean(BootstrapSaver['ModulationDepth'])
    PDResults['ModulationDepth']['std'] = np.std(BootstrapSaver['ModulationDepth'])
    
    PDResults['PD'] = {}
    PDResults['PD']['mean'] = np.mean(BootstrapSaver['PD'])
    PDResults['PD']['std'] = np.std(BootstrapSaver['PD'])
    
    return PDResults


class DirectionalPDFinder(MongoReadModule):
    def __init__(self,**kargs):
        TrialError = kargs['TrialError']
        del kargs['TrialError']
        super().__init__(**kargs)
        self.StatisticsMList =  [Seg for Seg in self.mldata.segments if Seg.description['TrialError']==TrialError]
    # Preprocessing method 
    def GetPDSpikeStatistics(self,**kargs):
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
            'UserDirection':List, Manual movement direction input

        Returns
        -------
        PD : Dict
            Containing the most important parameters of preferred direction model:
        PD containing the following parameters:
            'R2':fitting quality
            'BaseLine':interception of the PD model
            'ModulationDepth':productor of cosine function
            'PD':preferred direction

        '''
        
        self._PDInsertDict = kargs.copy()
        self._PDInsertDict['username'] = self.username
        self._PDInsertDict['TrialError'] = self.StatisticsMList[0].description['TrialError']
        kargs['TrialError'] = self.StatisticsMList[0].description['TrialError']
        
        FindInsert = self.mycol.find_one(self.vdi.encode(self._PDInsertDict))
        if FindInsert!=None: return
        BinWidth = kargs['t_right']-kargs['t_left'] #get BinWidth
        kargs['t_left'] = -BinWidth*kargs['cutoff']+kargs['t_left']
        kargs['t_right'] = BinWidth*kargs['cutoff']+kargs['t_right']
        kargs['sampling_period'] = BinWidth
        kargs['binsize'] = BinWidth
        kargs['nickname'] = 'SpikeStatistics4PD'
        #%% Preprocessing     
        self.debuger = kargs
        FiringRate = np.array(self.SpikeStatisticsReader(**kargs))
        
        #self.mycol.delete_many({'nickname':{'str':'SpikeStatistics4PD'}})
        
        if kargs['cutoff']!=0:
            FiringRate = FiringRate[:,:,kargs['cutoff']]
        FiringRate[FiringRate<0] = 0
        self.PopulationFiringRate = FiringRate
        
        if 'ContinousPD' in self._PDInsertDict['FunName']:
            return self.ContinousPD()
        if 'BootstrapContinousPD' in self._PDInsertDict['FunName']:
            return self.BootstrapContinousPD()
        if 'DiscreatePD' in self._PDInsertDict['FunName']:
            return self.DiscreatePD()
    
    def ReadoutPD(self):
        FiringRate = self.PopulationFiringRate.swapaxes(0,1).reshape((self.PopulationFiringRate.shape[0],-1))
        Variable = self._PDInsertDict['UserDirection'].swapaxes(0,1).reshape((self.PopulationFiringRate.shape[0],-1))
    #%% Fit PD model using continuous data
    def ContinousPD(self):
        '''
        After preprocessing, the statistic data can be fitted to continous movement direction

        Parameters
        ----------
        **kargs : {'MovementDirection':NumpyArray}
            A array related to 'StatisticsML'.

        Returns
        -------
        list
            Results and input parameters as index of Query.

        '''
        @SearchAndUpload
        def _FindPD(self):
            FiringRate = self.PopulationFiringRate
            Variable = np.array([[np.cos(i),np.sin(i)] for i in self._PDInsertDict['UserDirection']])
            data  = []
            for i in range(FiringRate.shape[-1]):
                PDResults = {}
                PDModel = linear_model.LinearRegression()
                PDModel.fit(Variable,FiringRate[:,i])
                PDResults['R2'] = PDModel.score(Variable,FiringRate[:,i])
                PDResults['BaseLine'] = PDModel.intercept_
                PDResults['ModulationDepth'] = np.linalg.norm(PDModel.coef_)
                PDResults['PD'] = np.math.atan2(PDModel.coef_[1],PDModel.coef_[0])
                self._PDInsertDict['ContinousPD'].append(PDResults.copy())
            return data
        return _FindPD(self,self._PDInsertDict)
            
    def BootstrapContinousPD(self):
        @SearchAndUpload
        def _FindPD(self):
            FiringRate = self.PopulationFiringRate 
            Variable = np.array([[np.cos(i),np.sin(i)] for i in self._PDInsertDict['UserDirection']])
            data = []
            for i in tqdm(range(FiringRate.shape[-1])):
                data.append(BootstrapPDModel(FiringRate[:,i],Variable,self._PDInsertDict['boostrap']))
            return data
        return _FindPD(self,self._PDInsertDict)
    
    def DiscreatePD(self):
        @SearchAndUpload
        def _FindPD(self):
            ConditionFiringRate = []
            ConditionVariable = []
            FiringRate = self.PopulationFiringRate
            
            ConditionVariable = list(np.unique(self._PDInsertDict['UserDirection']))
            ConditionFiringRate = [FiringRate[np.array(self._PDInsertDict['UserDirection'])==i].mean(0)\
                                   for i in np.unique(self._PDInsertDict['UserDirection'])]    
            ConditionVariable = np.array([[np.cos(i),np.sin(i)] for i in ConditionVariable])
            
            data = []
            PDModel = linear_model.LinearRegression()
            for i in range(ConditionFiringRate[0].shape[0]): 
                PDResults = {}
                PDModel.fit(ConditionVariable,np.array(ConditionFiringRate)[:,i])
                PDResults['R2'] = PDModel.score(ConditionVariable,np.array(ConditionFiringRate)[:,i])
                PDResults['BaseLine'] = PDModel.intercept_
                PDResults['ModulationDepth'] = np.linalg.norm(PDModel.coef_)
                PDResults['PD'] = np.math.atan2(PDModel.coef_[1],PDModel.coef_[0])
                data.append(PDResults)   
            return data
        return _FindPD(self,self._PDInsertDict)

if __name__ == '__main__':
    
    myclient = pymongo.MongoClient("mongodb://10.10.47.78:27017/", username='admin', password='cuilab324')
    mydb = myclient["Caesar_monkey"]
    #mkcollectiondir(CollectionList,Tempdir)
    CollectionList = mydb.list_collection_names()
    CollectionPD = {}
    
    for CollectionName in tqdm(CollectionList):
        
        if '3T' in CollectionName:
            continue
        
        PDFinder = DirectionalPDFinder(collection = CollectionName,
                                      SegName = 'Kilo_spike_train',
                                      Saverip="mongodb://10.10.47.78:27017/",
                                      db='Laplace_monkey_Sprobe',
                                      username='yongxiang',
                                      password='cuilab322',
                                      TrialError=0,
                                      LFP=False)
        
        Epoch = -2
        Object = 2

        CollectionPD[CollectionName] = {}
        for step in range(40):
            UserDirection = [np.math.atan2(*tuple(np.array(i.irregularlysampledsignals[2])[Epoch,Object,::-1]))\
                         for i in PDFinder.StatisticsMList]
            
            UserDirection = list(np.round(np.array(UserDirection)/(np.pi/3))*np.pi/3)[0:81]
            
            PD = PDFinder.GetPDSpikeStatistics(Statistics = 'time_histogram',
                                                t_left = -1000*pq.ms+step*50*pq.ms, 
                                                t_right = -1000*pq.ms+step*50*pq.ms+50*pq.ms,
                                                aligned_marker = 5,
                                                cutoff = 3,
                                                FunName = 'DiscreatePD',
                                                UserDirection = UserDirection,
                                                StatisticsML = list(range(100)))
            PD = PDFinder.PD
            PDFinder.mycol.delete_many({'nickname':{'str':'SpikeStatistics4PD'}})
    
    '''
    with open("/home/cuilab/Desktop/Caeser_TDC/CollectionPD.pickle",'wb') as f:
        pickle.dump(CollectionPD,f)'''
    
        