#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 21:41:57 2021

@author: cuilab
"""

import numpy as np
from neo import Segment,SpikeTrain
import quantities as pq
import pymongo

pqDict = {str(pq.Hz):pq.Hz,
          str(pq.s):pq.s}

def MongoDict2SpikeTrain(MongoDict,EndTrialCodeTime):
    SpikeTime = np.frombuffer(MongoDict['SpikeTimes'],dtype=np.int32)/(MongoDict['SampleRate']*pqDict[MongoDict['SampleRateUnit']])
    SpikeTime = SpikeTime.rescale(pq.s)
    spike_description = MongoDict['Description']
    
    
    train = SpikeTrain(times = SpikeTime,
                       t_stop = max(EndTrialCodeTime+10*pq.s,SpikeTime[-1]),
                       sampling_rate=MongoDict['SampleRate']*pq.Hz,                       
                       description=spike_description)
    
    return train

def MongoDict2Seg(mycol,SegName):
    mldata = mycol.find_one({'name':'BehaviorData'})['mldata']
    EndAbsoluteTrialStartTime = mldata['AbsoluteTrialStartTime'][-1]*pq.ms
    EndBehaviorCodes = mldata['BehavioralCodes']['CodeTimes'][-1][-1]*pq.ms
    EndTrialCodeTime = EndAbsoluteTrialStartTime+EndBehaviorCodes
    Seg = Segment(name=SegName)
    
    assert mycol.find_one({'SegName':SegName})!=None,'Data not existed'        
        
    for MongoDict in mycol.find({'SegName':SegName,'name':'SpikeTrain'}):
        train = MongoDict2SpikeTrain(MongoDict,EndTrialCodeTime)
        Seg.spiketrains.append(train)
        
    return Seg
