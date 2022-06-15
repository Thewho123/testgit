#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:00:06 2021

@author: cuilab
"""


import numpy as np
from ..MongoReader import MongoReadModule
from ..DataPreprocessor2 import GenerateQuery
#from joblib import Parallel,delayed
from tqdm import tqdm
import quantities as pq
#from elephant.kernels import GaussianKernel
import pymongo
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

class RepresentationalParametersCurveFitter(MongoReadModule):
    
    # Preprocessing method 
    def FindCurve(self,**kargs):
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
        
        InsertDict = {}
        GenerateQuery(InsertDict,kargs)
        InsertDict['name'] = 'Curve'
        InsertDict['username'] = self.username
        GenerateQuery(InsertDict,kargs)
        
        if self.mycol.find_one(InsertDict)!=None: return [i for i in self.mycol.find(InsertDict)]
        
        BinWidth = kargs['t_right']-kargs['t_left'] #get BinWidth
        #%% Preprocessing
        if 'instantaneous_rate' in kargs['Statistics']:
            FiringRate = self.SpikeFiringRate(Statistics = 'instantaneous_rate',
                                              sampling_period = BinWidth, 
                                              t_left = -BinWidth*kargs['cutoff']+kargs['t_left'], 
                                              t_right = BinWidth*kargs['cutoff']+kargs['t_right'],
                                              kernel = kargs['kernel'],
                                              aligned_marker = kargs['aligned_marker'],
                                              TrialError=kargs['TrialError'],
                                              StatisticsML = kargs['StatisticsML'])
            
        if 'time_histogram' in kargs['Statistics']:
            FiringRate = self.SpikeBin(Statistics = 'time_histogram',
                                       binsize = BinWidth, 
                                       t_left = -BinWidth*kargs['cutoff']+kargs['t_left'], 
                                       t_right = BinWidth*kargs['cutoff']+kargs['t_right'],
                                       aligned_marker = kargs['aligned_marker'],
                                       TrialError=kargs['TrialError'],
                                       StatisticsML = kargs['StatisticsML'])
            
        if kargs['cutoff']!=0:
            FiringRate = FiringRate[:,kargs['cutoff'],:]
        FiringRate[FiringRate<0] = 0
        self.PopulationFiringRate = FiringRate
        if 'ContinousCurve' in kargs['FunName']:
            Curve = self.ContinousCurve(InsertDict,**kargs)
            self.DeleteSelfQuery()
            return Curve
        
        if 'DiscreateCurve' in kargs['FunName']:
            Curve = self.DiscreateCurve(InsertDict,**kargs)
            self.DeleteSelfQuery()
            return Curve
        
    #%% Fit PD model using continuous data
    def ContinousCurve(self,InsertDict,**kargs):
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
        
        if self.mycol.find_one(InsertDict)!=None: return [i for i in self.mycol.find(InsertDict)]
        FiringRate = self.PopulationFiringRate
        Variable = kargs['Variable']
        
        for i in range(FiringRate.shape[-1]):
            popt,pcov = curve_fit(kargs.get('model'),
                                  xdata = Variable,
                                  ydata = FiringRate[:,i],
                                  bounds= kargs['bounds'],
                                  maxfev= kargs['maxfev'])
            
            ypred = [kargs.get('model')(xp,*popt) for xp in Variable]
            CurveFitResults = {}
            CurveFitResults['R2'] = r2_score(FiringRate[:,i],np.array(ypred))
            CurveFitResults['FitVariable'] = {}
            CurveFitResults['FitVariable']['value'] = popt
            CurveFitResults['FitVariable']['name'] = InsertDict['VariableName']
            InsertDict[self.ElectrodeMap[kargs['Statistics']][i]] = CurveFitResults.copy()
        
        self.mycol.insert_one(InsertDict.copy())
        return [i for i in self.mycol.find(InsertDict)]
    
    def DiscreateCurve(self,InsertDict,**kargs):
        
        ConditionFiringRate = []
        ConditionVariable = []
        FiringRate = self.PopulationFiringRate
        Variable = kargs['Variable']
            
        for i in range(ConditionFiringRate[0].shape[0]): 
            ConditionFiringRate = []
            ConditionVariable = []
            CurveFitResults = {}
            for ind,j in enumerate(np.unique(kargs['Group'])):
                ConditionFiringRate.append(np.mean(FiringRate[kargs['Group']==j,i]))
                ConditionVariable.append(Variable[j])
            popt,pcov = curve_fit(kargs.get('model'),
                                  xdata = np.array(ConditionVariable),
                                  ydata = np.array(ConditionFiringRate),
                                  bounds= kargs['bounds'],
                                  maxfev= kargs['maxfev'])
            
            ypred = [kargs.get('model')(xp,*popt) for xp in ConditionVariable]
            CurveFitResults['R2'] = r2_score(np.array(ConditionFiringRate),np.array(ypred))
            CurveFitResults['FitVariable'] = {}
            CurveFitResults['FitVariable']['value'] = popt
            CurveFitResults['FitVariable']['name'] = InsertDict['VariableName']
            InsertDict[self.ElectrodeMap[kargs['Statistics']][i]] = CurveFitResults
        
        self.mycol.insert_one(InsertDict.copy())
        return [i for i in self.mycol.find(InsertDict)]

if __name__ == '__main__':
    
    myclient = pymongo.MongoClient("mongodb://10.10.47.78:27017/", username='admin', password='cuilab324')
    mydb = myclient["Caesar_monkey"]
    #mkcollectiondir(CollectionList,Tempdir)
    CollectionList = mydb.list_collection_names()
    CollectionPD = {}
    
    for CollectionName in tqdm(CollectionList):
        
        if '3T' in CollectionName:
            continue
        
        PDFinder = RepresentationalParametersCurveFitter(collection = CollectionName,
                                                           SegName = 'FinalEnsemble_spike_train',
                                                           Saverip="mongodb://10.10.47.78:27017/",
                                                           db='Caesar_monkey',
                                                           username='yongxiang',
                                                           password='cuilab322')
    
    '''
    with open("/home/cuilab/Desktop/Caeser_TDC/CollectionPD.pickle",'wb') as f:
        pickle.dump(CollectionPD,f)'''
    
        