#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:57:01 2021

@author: cuilab
"""

#from MongoNeo.DataPreprocessor_multi import DataPreprocessor
from .DataPreprocessor2 import DataPreprocessor

class MongoReadModule(DataPreprocessor):            
    
    def SpikeStatisticsReader(self,**kargs):
        '''
        Convert bytes array to numpy,
        inherited from DataPreprocessor module
        used for neural instantaneous rate finding
        
        Parameters
        ----------
        **kargs : {Statistics : 'instantaneous_rate',
                   sampling_period : time bin width, 
                   t_left = left border to aligned marker, 
                   t_right = right border to aligned marker,
                   kernel = kernel type and parameters of elephant.kernels,
                   aligned_marker = choosed aligned marker in a trial,
                   TrialError=trial error marker of behavior marker,
                   StatisticsML = behavior data list as label to capture data, it can be set to None to capture all data}

        Returns
        -------
        TYPE
            np.array of spike firing with trial x time x cell shape.

        '''        
        if 'DeleteQuery' in kargs.keys():
            DeleteQuery = kargs['DeleteQuery']
            del kargs['DeleteQuery']
        else:
            DeleteQuery = False
            
        self.SpikeStatisticsList = self.SpikeStatistics(**kargs)
    
        if DeleteQuery:
            self.DeleteSpikeStatistics()       
        return self.SpikeStatisticsList
    def DeleteSpikeStatistics(self):
        self.mycol.delete_one(self.vdi.encode(self.kargs))
                
                