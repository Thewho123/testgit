#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 22:03:01 2021

@author: cuilab
"""

from MongoNeo.user_layer.pipeline_template.neural_data.br_share.Sorter2Dict.Sorter2Dict import Sorter2Dict

def NullLabel(label:str):
    def CheckNull(func):
        def warpped_function(InputData,userElement):
            if isinstance(userElement[label],str):
                if userElement[label]=='null':
                    InputData[label] = 'null'
                    return
            func(InputData,userElement)
        return warpped_function
    return CheckNull

def BuilddataOperationMap()->dict:
    '''
    A map to handle .mat data given by monkeylogic

    Returns
    -------
    dict
        A dict contains column labels and function for operation.

    '''
    
    # initialize the map (dict)
    dataOperationMap = {}
    
    @NullLabel('RecordingSystemEvent')
    def RecordingSystemEventOperation(InputData,userElement):
        InputData['RecordingSystemEvent']['event'] = userElement['RecordingSystemEvent']
        
    @NullLabel('LFP')
    def LFPOperation(InputData,userElement):
        InputData['LFP']['ana'] = userElement['LFP']
    
    @NullLabel('Spike')
    def SpikeOperation(InputData,userElement):
        InputData['Spike'] = userElement['Spike']
    
    @NullLabel('name')
    def nameOperation(InputData,userElement):
        InputData['name'] = userElement['name']
    
    # Build the map
    dataOperationMap['RecordingSystemEvent'] = RecordingSystemEventOperation
    dataOperationMap['LFP'] = LFPOperation
    dataOperationMap['Spike'] = SpikeOperation
    dataOperationMap['name'] = nameOperation
    
    return dataOperationMap
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
