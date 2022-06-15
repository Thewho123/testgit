#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:12:28 2020

@author: cuilab
"""

import joblib
from .HD2Dict import HDSorter2Dict
from .Iron2Dict import IronSorter2Dict
from .MS2Dict import MSSorter2Dict
from .tdc2Dict import tdcSorter2Dict
from .kilo2Dict import kiloSorter2Dict
import quantities as pq

def SpykingcircusReadFun(sorter_dir, ChnNum = None, wavedict = {},
                RecordingParam = None,sampling_rate = 30000*pq.Hz):
    sc_dir = RecordingParam
    with open(sc_dir,"rb") as f:
        Seg = joblib.load(f)
    return Seg

def Herdingspike2ReadFun(sorter_dir, ChnNum = None, wavedict = {},
                RecordingParam = None,sampling_rate = 30000*pq.Hz):
    hs2_dir = RecordingParam
    with open(hs2_dir,"rb") as f:
        Seg = joblib.load(f)
    return Seg

Sorter2Dict = {'tdc_spike_train':tdcSorter2Dict.Sorter2Dict,
               'Kilo_spike_train':kiloSorter2Dict.Sorter2Dict,
               'Spykingcircus_spike_train':SpykingcircusReadFun,
               'Herdingspike2_spike_train':Herdingspike2ReadFun,
               'IronClust_spike_train':IronSorter2Dict.Sorter2Dict,
               'HDSort_spike_train':HDSorter2Dict.Sorter2Dict,
               'Mountain4_spike_train':MSSorter2Dict.Sorter2Dict}

    
    
    
    
    
    
    
    
    
    
    
    
    
    