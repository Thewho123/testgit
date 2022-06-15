#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 14:43:43 2021

@author: cuilab
"""
import numpy as np
from MongoNeo.DataPreprocessor_multi import DataPreprocessor
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

class RepresentationalParametersCurveFitter(DataPreprocessor):
    def ContinousCurveFitter(self,**kargs):
        InsertDict = {}
        BinWidth = kargs['t_right']-kargs['t_left']
        InsertDict['name'] = 'RepresentationalParametersCurveFitter'
        InsertDict['FunName'] = 'ContinousCurveFitter'
        
        for PDDict in kargs.keys():
            try:
                Magnitude = float(kargs[PDDict].magnitude)
                unit = kargs[PDDict].units
                InsertDict[PDDict] = {}
                InsertDict[PDDict]['magnitude'] = Magnitude
                InsertDict[PDDict]['unit'] = str(unit)
                continue
            except:
                pass
            try:
                InsertDict[PDDict] = float(kargs[PDDict])
                continue
            except:
                pass
            
            try:
                InsertDict[PDDict] = str(kargs[PDDict])
                continue
            except:
                pass
            InsertDict[PDDict] = kargs[PDDict]
            
        if self.mycol.find_one(InsertDict)!=None: return [i for i in self.mycol.find(InsertDict)]
        
        if 'instantaneous_rate' in kargs['Statistics']:
            FiringRate = self.SpikeFiringRate(Statistics = 'instantaneous_rate',
                                              sampling_period = BinWidth, 
                                              t_left = -BinWidth*kargs['cutoff'], 
                                              t_right = BinWidth*kargs['cutoff'],
                                              kernel = kargs['kernel'],
                                              aligned_marker = kargs['aligned_marker'],
                                              TrialError=kargs['TrialError'],
                                              StatisticsML = kargs['StatisticsML'])
        if 'time_histogram' in kargs['Statistics']:
            FiringRate = self.SpikeBin(Statistics = 'time_histogram',
                                       binsize = BinWidth, 
                                       t_left = -BinWidth*kargs['cutoff'], 
                                       t_right = BinWidth*kargs['cutoff'],
                                       aligned_marker = kargs['aligned_marker'],
                                       TrialError=kargs['TrialError'])
        if kargs['cutoff']!=0:
            FiringRate = FiringRate[:,kargs['cutoff'],:]
            
        for i in range(FiringRate.shape[-1]): 
            Variable = kargs['Variable']
            
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
            InsertDict[self.ElectrodeMap[kargs['Statistics']][i]] = CurveFitResults
            
        InsertResult = self.mycol.insert_one(InsertDict.copy())
        return [i for i in self.mycol.find(InsertDict)]
    
    def DiscreateCurveFitter(self,**kargs):
        InsertDict = {}
        BinWidth = kargs['t_right']-kargs['t_left']
        InsertDict['name'] = 'RepresentationalParametersCurveFitter'
        InsertDict['FunName'] = 'DiscreateCurveFitter'
        
        for PDDict in kargs.keys():
            try:
                Magnitude = float(kargs[PDDict].magnitude)
                unit = kargs[PDDict].units
                InsertDict[PDDict] = {}
                InsertDict[PDDict]['magnitude'] = Magnitude
                InsertDict[PDDict]['unit'] = str(unit)
                continue
            except:
                pass
            try:
                InsertDict[PDDict] = float(kargs[PDDict])
                continue
            except:
                pass
            
            try:
                InsertDict[PDDict] = str(kargs[PDDict])
                continue
            except:
                pass
            InsertDict[PDDict] = kargs[PDDict]
            
        if self.mycol.find_one(InsertDict)!=None: return [i for i in self.mycol.find(InsertDict)]
        
        if 'instantaneous_rate' in kargs['Statistics']:
            FiringRate = self.SpikeFiringRate(Statistics = 'instantaneous_rate',
                                              sampling_period = BinWidth, 
                                              t_left = -BinWidth*kargs['cutoff'], 
                                              t_right = BinWidth*kargs['cutoff'],
                                              kernel = kargs['kernel'],
                                              aligned_marker = kargs['aligned_marker'],
                                              TrialError=kargs['TrialError'],
                                              StatisticsML = kargs['StatisticsML'])
        if 'time_histogram' in kargs['Statistics']:
            FiringRate = self.SpikeBin(Statistics = 'time_histogram',
                                       binsize = BinWidth, 
                                       t_left = -BinWidth*kargs['cutoff'], 
                                       t_right = BinWidth*kargs['cutoff'],
                                       aligned_marker = kargs['aligned_marker'],
                                       TrialError=kargs['TrialError'])
        if kargs['cutoff']!=0:
            FiringRate = FiringRate[:,kargs['cutoff'],:]
        
        for i in range(FiringRate.shape[-1]): 
            Variable = kargs['Variable']
            ConditionFiringRate = []
            ConditionVariable = []
            CurveFitResults = {}
            for j in np.unique(Variable):
                ConditionFiringRate.append(np.mean(FiringRate[Variable==j,i]))
                ConditionVariable.append(j)
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
        
        InsertResult = self.mycol.insert_one(InsertDict.copy())
        return [i for i in self.mycol.find(InsertDict)]