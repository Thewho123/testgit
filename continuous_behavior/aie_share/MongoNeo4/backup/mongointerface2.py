#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:12:33 2021

@author: cuilab
"""


import os
import numpy as np
import quantities as pq
import glob
import json

def SegSpikeTrain2Dict(st,name,SampleRate=30000):
    SpikeTrainDict = {'units':str(st.units),
                      'SegName':name,
                      'SampleRate':SampleRate,
                      'SampleRateUnit':str(pq.Hz),
                      'name':'SpikeTrain'}
    SampleRate = SampleRate*pq.Hz
    DescriptionDict = {}
    if st.description!=None:
        DescriptionDict = st.description.copy()
        
        if 'par' in DescriptionDict.keys():
            del DescriptionDict['par']
        
        if 'mean_waveform' in DescriptionDict.keys():
            DescriptionDict['mean_waveform'] = list(DescriptionDict['mean_waveform'].astype(float))
        
        for key in DescriptionDict.keys():
            try:
                DescriptionDict[key] = DescriptionDict[key].astype(float)
            except:
                try:
                    DescriptionDict[key] = float(DescriptionDict[key])
                except:
                    pass
        
    SpikeTrains = st.times.rescale(pq.s)
    SpikeTrains = SpikeTrains*SampleRate
    SpikeTrainDict.update({'Description':DescriptionDict})
    SpikeTrainDict.update({'SpikeTimes':(SpikeTrains.magnitude).astype(np.int32).tobytes()})
    SpikeTrainDict.update({'SpikeTimesDtype':str(np.int32)})
    
    return SpikeTrainDict

def ResetTdcInfo(dataio_dir):
    raw_file = glob.glob(os.path.join(dataio_dir+'/mda','*.raw'))
    try:
        info_path = glob.glob(os.path.join(dataio_dir,'*.json'))[0]
    except IndexError:
        return
    
    with open(info_path,'r',encoding='utf-8')as fp:
        json_data = json.load(fp)
    
    json_data['datasource_kargs']['filenames'] = raw_file
    
    os.remove(info_path)
    with open(info_path,'w',encoding='utf-8') as json_file:
        json.dump(json_data,json_file)

                    
                
            
        
