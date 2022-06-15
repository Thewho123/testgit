#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:12:28 2020

@author: cuilab
"""

import quantities as pq
import numpy as np
from MongoNeo.user_layer.dict_to_neo import templat_neo
from .BaseSorter2Dict import BaseSorter2Dict
import hdf5storage

def HDSortShare(ParseResults, wavedict ,
                 RecordingParam, sampling_rate):
    SpikeDict = {}
    SpikeDict['name'] = 'HDSort_spike_train'
    for n,i in enumerate(ParseResults):
        hdFile = i['detectionChannel']
        for j in np.unique(hdFile):
            unit = {'spk':{}}
            unit['spk'] = templat_neo['spk'].copy()
            
            hdChFile = i['spikeTrain'][hdFile==j]
            if hdChFile.shape[0]<5000:continue
            spike_description = {'Clu':float(n),
                                 'Chn':float(j-1),
                                 'mean_waveform':wavedict[n] if n in wavedict else str(None)}
            
            if str(spike_description['Chn']) not in SpikeDict:
                SpikeDict[str(spike_description['Chn'])] = {}
        
            if str(spike_description['Clu']) not in SpikeDict[str(spike_description['Chn'])]:
                SpikeDict[str(spike_description['Chn'])][str(spike_description['Clu'])] = {}
                
            unit['spk']['times'] = hdChFile/sampling_rate
            unit['spk']['t_stop'] = hdChFile[-1]/sampling_rate
            unit['spk']['sampling_rate'] = sampling_rate
            unit['spk']['description'] = spike_description
            unit['spk']['t_start'] = 0*unit['spk']['times'].units      
            SpikeDict[str(spike_description['Chn'])][str(spike_description['Clu'])] = unit
            
    return SpikeDict

class HDSorter2Dict(BaseSorter2Dict):
    def _ParseSorterResults(self, sorter_dir):
        self._sorter_dir = sorter_dir
        self._ParseResults = hdf5storage.loadmat(sorter_dir)['Units'].squeeze()
    
    def _ExtractMeanWaveform(self, **waveform_args):
        return {}
    
    @classmethod        
    def Sorter2Dict(cls, sorter_dir, wavedict = {},
                    RecordingParam = None,sampling_rate = 30000*pq.Hz,**waveform_args):
        self = cls()
        self._ParseSorterResults(sorter_dir)
        
        if len(wavedict) == 0:
            wavedict = self._ExtractMeanWaveform(**waveform_args)
            
        return HDSortShare(self._ParseResults, wavedict = wavedict,
                           RecordingParam = RecordingParam,sampling_rate = sampling_rate)
    
    
    
    
    
    
    
    
    
    
    
    
    
    