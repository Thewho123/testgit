# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:04:22 2022

@author: syj
"""
# br data如何从0.4转成mongoneo 0.5
import os
import copy
import glob
import argparse
import joblib
import quantities as pq
from MongoNeo.user_layer.dict_to_neo import templat_neo
from brpylib import NsxFile
from neo import io
import yaml
from user_input_entry import BRShare as bs
from MongoNeo4.MongoReader import MongoReadModule
#%% parse the input arguments
parser = argparse.ArgumentParser(argument_default=None)
parser.add_argument("-f", "--file", type=str,
                    help="file of neural data")
parser.add_argument("-o", "--output", type=str,
                    help="the path to save the results")
args = parser.parse_args()
raw_dirname = args.file
save_dir=args.output
# raw_dirname = '/home/cuilab/Desktop/warmdata/CaeserData/BehaviorData/tdc_caeser202010173T001_Frank_peeler_wtw'
# save_dir='/home/cuilab/Desktop/hotdata/data_integration_pipeline/sun_pipeline/neural_data/br_share/test.db'

#%% parameter for MongoNeo4
CollectionName=raw_dirname.split('/')[-1]
if 'caesar' in CollectionName or 'Caesar' in CollectionName or \
    'caeser' in CollectionName or 'Caeser' in CollectionName:
    MONKEY_DB='Caesar_monkey_array'
elif 'Gauss' in CollectionName or 'gauss' in CollectionName or \
    'caeser' in CollectionName or 'Caeser' in CollectionName:
    MONKEY_DB='Gauss_monkey_array'
elif 'Poisson' in CollectionName or 'poisson' in CollectionName or\
    'Piosson' in CollectionName or 'piosson' in CollectionName:
    MONKEY_DB='Poisson_monkey_array'
else:
    print("Please input monkeyname in pipiline!")

#%% load template
FILEPATH = os.path.dirname(os.path.abspath(__file__))#获得当前脚本的完整路径
with open(os.path.join(FILEPATH,'template.yml'), 'r',encoding="utf-8") as f:
    Template = yaml.safe_load(f)
Template['LFP'] = templat_neo['ana']
Template['RecordingSystemEvent'] = templat_neo['event']
Template['Spike'] = {'SorterName' : {},'kargs' : {}}

InputList = []
#%% convert LFP
InputData = copy.deepcopy(Template)
InputData['RecordingSystemEvent'] = 'null'
InputData['Spike'] = 'null'
ns3File = glob.glob(raw_dirname+'/*.ns3')[0]
ns3_file = NsxFile(ns3File)
LFPRecordingParam = ns3_file.extended_headers
LFPData = ns3_file.getdata()
InputData['LFP']['sampling_rate'] = LFPData['samp_per_s']*pq.Hz
InputData['LFP']['signal'] = LFPData['data']*pq.uV
InputData['LFP']['t_start'] = 0*pq.s
InputData['LFP']['description'] = LFPRecordingParam
InputData['name'] = 'LFP'
InputList.append(InputData)
ns3_file.close()

#%% convert RecordingSystemEvent
InputData = copy.deepcopy(Template)
InputData['LFP'] = 'null'
InputData['Spike'] = 'null'
nsFile = glob.glob(raw_dirname+'/*.ns6')[0]
nsx_file = NsxFile(nsFile)
RawRecordingParam = nsx_file.extended_headers
sampling_rate = nsx_file.basic_header['TimeStampResolution']*pq.Hz
nsx_file.close()
nev_dir = glob.glob(os.path.join(raw_dirname,'nev')+'/*.nev')[0]
blk_event_marker = io.BlackrockIO(nev_dir)
InputData['RecordingSystemEvent']['times'] = \
    blk_event_marker.get_event_timestamps()[0]/sampling_rate.magnitude*pq.sec
InputData['RecordingSystemEvent']['labels'] = \
    [float(i) for i in blk_event_marker.get_event_timestamps()[2]]
InputData['name'] = 'RecordingSystemEvent'
InputList.append(InputData)

#%% All auto-sorting method name
sorter_list=['Kilo_spike_train','tdc_spike_train','Spykingcircus_spike_train',
        'Herdingspike2_spike_train','IronClust_spike_train','HDSort_spike_train'
        ,'FinalEnsemble_spike_train','Ensemble_spike_train']

#%% load data from MongoNeo4
MongoReader = MongoReadModule(collection = CollectionName,
                          SegName = sorter_list,
                          Saverip="mongodb://10.10.47.78:27017/",
                          db=MONKEY_DB,
                          username='yongxiang',
                          password='cuilab322',
                          LFP=False)

for sorterDB in sorter_list:
    InputData = copy.deepcopy(Template)
    InputData['LFP'] = 'null'
    InputData['RecordingSystemEvent'] = 'null'
    InputData['Spike'] = {}
    for sorter in MongoReader.block_br.segments:
        if sorter.name=='tdc_spike_train':
            tdc=sorter
            break
    InputData['name'] = str.lower(sorterDB)
    SpikeDict={}
    SpikeDict['name'] = str.lower(sorterDB)
    for i,sorter in enumerate(MongoReader.block_br.segments):
        if sorter.name!=sorterDB:
            continue
        sorter_spiketrain=MongoReader.block_br.segments[i].spiketrains
        tdc=tdc.spiketrains
        for (sorter_unit,tdc_unit) in zip(sorter_spiketrain,tdc):
            spike_description = {'clu':tdc_unit.description['clu'],
                              'snr':tdc_unit.description['snr'],
                              'chn':tdc_unit.description['group'],
                              'mean_waveform':tdc_unit.description['mean_waveform']}
            unit = {'spk':{}}#One unit represents one neurons
            unit['spk'] = templat_neo['spk'].copy()
            unit['spk']['times'] = sorter_unit.times #
            unit['spk']['t_stop'] = sorter_unit.t_stop
            unit['spk']['sampling_rate'] = sorter_unit.sampling_rate
            unit['spk']['t_start'] = sorter_unit.t_start
            unit['spk']['description'] = spike_description
            if str(spike_description['chn']) not in SpikeDict:
                SpikeDict[str(spike_description['chn'])] = {}
            if str(spike_description['clu']) not in SpikeDict[str(spike_description['chn'])]:
                SpikeDict[str(spike_description['chn'])][str(spike_description['clu'])] = {}
            SpikeDict[str(spike_description['chn'])][str(spike_description['clu'])] = unit
        InputData['Spike'] =SpikeDict
        break
    if sorter.name in sorter_list:
        InputList.append(InputData)
neuralblock = bs.data_input(user_data = InputList,index = 1)
#%% save to appointed path
if save_dir is not None:
    joblib.dump(neuralblock, filename=save_dir)