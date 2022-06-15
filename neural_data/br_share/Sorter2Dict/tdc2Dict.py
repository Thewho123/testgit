#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:12:28 2020

@author: cuilab
"""

from tridesclous import DataIO
import quantities as pq
import numpy as np
from tqdm import tqdm
from MongoNeo.user_layer.dict_to_neo import templat_neo
from .BaseSorter2Dict import BaseSorter2Dict

def set_spike_segment(dataio_br, Chn, SpikeDict, RecordingParam, dataio_dir):
    #%% align segments spike data
    memmap = dataio_br.get_spikes(chan_grp=Chn,seg_num = 0)
    spike_time = memmap['index'].copy()
    cluster_label = memmap['cluster_label'].copy()
    #print('cluster:',cluster_num)
    cluster_num = list(dataio_br.load_catalogue(chan_grp=Chn)['cluster_labels'])
    processed_signals = dataio_br.arrays[Chn][0].get('processed_signals')
    
    #%% set spiketrains list  
    for clu in cluster_num:
        unit = {'spk':{}}
        unit['spk'] = templat_neo['spk'].copy()
        
        spike_index = spike_time[cluster_label==clu]
        if len(spike_index)==0:
            continue
        Select = (spike_index<(len(processed_signals)-32)) * (spike_index>16)
        spike_index = spike_index[Select]

        cluster_spike_times = (spike_index/dataio_br.sample_rate)*pq.s

        #QuatoIndex = int(processed_signals.shape[0]/4)
        peak_amplitude = processed_signals[spike_index]
        mad = np.median(np.abs(processed_signals-np.median(processed_signals)))  
        snr = np.mean(np.abs(peak_amplitude))/(mad*1.4826) 
        spike_description = {'Clu':str(clu),
                             'snr':snr,
                             'Chn':Chn,
                             'mean_waveform':np.array([processed_signals[(index-16):(index+32)]\
                                               for index in spike_index]).squeeze().mean(0)}
        
        if isinstance(RecordingParam,dict):
            spike_description.update(RecordingParam[Chn])
            
        if str(spike_description['Chn']) not in SpikeDict:
            SpikeDict[str(spike_description['Chn'])] = {}
        
        if str(spike_description['Clu']) not in SpikeDict[str(spike_description['Chn'])]:
            SpikeDict[str(spike_description['Chn'])][str(spike_description['Clu'])] = {}
            
        unit['spk']['times'] = cluster_spike_times
        unit['spk']['t_stop'] = cluster_spike_times[-1]
        unit['spk']['sampling_rate'] = dataio_br.sample_rate*pq.Hz
        unit['spk']['t_start'] = 0*cluster_spike_times.units
        unit['spk']['description'] = spike_description      
        SpikeDict[str(spike_description['Chn'])][str(spike_description['Clu'])] = unit
        
    dataio_br.arrays[Chn][0].detach_array('spikes')
    dataio_br.arrays[Chn][0].detach_array('processed_signals')
    
class tdcSorter2Dict(BaseSorter2Dict):
    def _ParseSorterResults(self, sorter_dir):
        self._dataio_br = DataIO(dirname=sorter_dir)        
        self._sorter_dir = sorter_dir
    
    def _ExtractMeanWaveform(self):
        pass
    
    @classmethod       
    def Sorter2Dict(cls, sorter_dir, wavedict = {},
                    RecordingParam = None,sampling_rate = 30000*pq.Hz,**waveform_args):
        self = cls()
        self._ParseSorterResults(sorter_dir)
        
        SpikeDict = {}
        SpikeDict['name'] = 'tdc_spike_train'
        dataio_br = DataIO(dirname=sorter_dir)
        for Chn in tqdm(self._dataio_br.channel_groups.keys()):
            set_spike_segment(dataio_br, Chn, SpikeDict, wavedict, RecordingParam, sorter_dir)
            
        return SpikeDict
    
    
    
    
    
    
    
    
    
    
    
    