#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:12:28 2020

@author: cuilab
"""
import phy
import quantities as pq
import os
import glob
import numpy as np
from MongoNeo.user_layer.dict_to_neo import templat_neo
from .BaseSorter2Dict import BaseSorter2Dict
from phylib.io.model import TemplateModel

def ReadKiloSort(kilo_dir):
    KiloSortResult = {}
    import pandas as pd
    if len(glob.glob(os.path.join(kilo_dir,'cluster_info*')))!=0:
        info = pd.read_csv(glob.glob(os.path.join(kilo_dir,'cluster_info*'))[0], delimiter='\t')
        KiloSortResult['cluster'] = np.load(glob.glob(os.path.join(kilo_dir,'spike_clusters*'))[0])
        KiloSortResult['time'] = np.load(glob.glob(os.path.join(kilo_dir,'spike_times*'))[0])
        KiloSortResult['id'] = list(info['id'] if 'id' in info else info['cluster_id'])
        KiloSortResult['group'] = list(info['group']) if not pd.isna(info['group'][0]) else list(info['KSLabel'])
        KiloSortResult['ch'] = list(info['ch'])
    else:
        model = TemplateModel(dir_path = kilo_dir)
        KiloSortResult['cluster'] = np.load(glob.glob(os.path.join(kilo_dir,'spike_clusters*'))[0])
        KiloSortResult['time'] = np.load(glob.glob(os.path.join(kilo_dir,'spike_times*'))[0])
        KiloSortResult['id'] = list(range(len(model.metadata['KSLabel'])))
        KiloSortResult['group'] = model.metadata['KSLabel']
        KiloSortResult['ch'] = list(range(len(model.metadata['KSLabel'])))
    return KiloSortResult

def Kiloshare(KiloSortResult,wavedict = {},RecordingParam = None,sampling_rate = 30000*pq.Hz):    
    SpikeDict = {}
    SpikeDict['name'] = 'Kilo_spike_train'
    
    for ind,i in enumerate(KiloSortResult['id']):
        unit = {'spk':{}}
        unit['spk'] = templat_neo['spk'].copy()
        
        if not isinstance(KiloSortResult['group'][ind], str):
            continue
        
        if 'good' not in KiloSortResult['group'][ind]:
            continue
        
        mean_waveform = wavedict[i] if i in wavedict else str(None)              
                
        spike_description = {'Clu':float(i),
                             'Chn':float(KiloSortResult['ch'][ind]),
                             'mean_waveform':mean_waveform}
        
        if str(spike_description['Chn']) not in SpikeDict:
            SpikeDict[str(spike_description['Chn'])] = {}
        
        if str(spike_description['Clu']) not in SpikeDict[str(spike_description['Chn'])]:
            SpikeDict[str(spike_description['Chn'])][str(spike_description['Clu'])] = {}
        
        if isinstance(RecordingParam,dict):
            spike_description.update(RecordingParam[KiloSortResult['ch'][ind]])

        KiloSpike = KiloSortResult['time'][KiloSortResult['cluster']==i].squeeze()/sampling_rate
        
        unit['spk']['times'] = KiloSpike
        unit['spk']['t_stop'] = KiloSpike[-1]
        unit['spk']['t_start'] = 0*KiloSpike.units
        unit['spk']['sampling_rate'] = sampling_rate
        unit['spk']['description'] = spike_description
        SpikeDict[str(spike_description['Chn'])][str(spike_description['Clu'])] = unit
    
    return SpikeDict

class kiloSorter2Dict(BaseSorter2Dict):
    def _ParseSorterResults(self, sorter_dir):
        self._sorter_dir = sorter_dir
        self._ParseResults = ReadKiloSort(sorter_dir)
    
    def _ExtractMeanWaveform(self, **waveform_args):
        temp_wh_path = os.path.join(self._sorter_dir,'temp_wh.dat')
        if len(waveform_args) or not os.path.exists(temp_wh_path):
            return {}
        
        fp = np.memmap(temp_wh_path, dtype='int16', mode='r',shape=(-1,waveform_args['ChnNum']))
        wavedict = {}
        for ind,i in enumerate(self._ParseResults['id']):
            RawDataArray = fp[:,self._ParseResults['ch'][ind]]
            waveformsList = [RawDataArray[(int(index)-16):(int(index)+32)]\
                                                  for index in self._ParseResults['time'][self._ParseResults['cluster']==i].squeeze()]
            waveforms=np.array(waveformsList[0:-2])
            wavedict[ind] = {}
            wavedict[ind][i] = waveforms.squeeze().mean(0)
        return wavedict
    
    @classmethod
    def Sorter2Dict(cls, sorter_dir, wavedict = {},
                    RecordingParam = None,sampling_rate = 30000*pq.Hz, **waveform_args):
        self = cls()
        self._ParseSorterResults(sorter_dir)
        
        if len(wavedict) == 0:
            wavedict = self._ExtractMeanWaveform(**waveform_args)
            
        return Kiloshare(self._ParseResults,wavedict = wavedict,
                         RecordingParam = RecordingParam,sampling_rate = sampling_rate)
    
    
    
    
    
    
    
    
    
    
    
    
    
    