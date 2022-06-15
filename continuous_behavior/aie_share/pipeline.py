#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Created on Wed Jun  1 14:46:20 2022

@author: cuilab
"""
import os
import argparse
import copy
import yaml
import quantities as pq
import joblib
import numpy as np
from MongoNeo.user_layer.dict_to_neo import templat_neo
from user_input_entry import AIEShare as As
from MongoNeo4.MongoReader import MongoReadModule
#%% parse the input arguments
parser = argparse.ArgumentParser(argument_default=None)
parser.add_argument("-f", "--file", type=str,
                    help="file of behevior data")
parser.add_argument("-o", "--output", type=str,
                    help="the path to save the results")
args = parser.parse_args()
raw_dirname = args.file
#%% load data from MongoNeo4
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
    print("Please input monkeyname in pipeline!")   
MongoReader = MongoReadModule(collection = CollectionName,
                          SegName = [],
                          Saverip="mongodb://10.10.47.78:27017/",
                          db=MONKEY_DB,
                          username='yongxiang',
                          password='cuilab322',
                          LFP=False)
#%% load template
FILEPATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(FILEPATH,'template.yml'),encoding=('utf-8')) as f:
    Template = yaml.safe_load(f)

InputList = []

#%% convert IrregularSampledData:Motion Capture
InputData = copy.deepcopy(Template)
InputData['AnalogData'] = 'null'
InputData['Event'] = 'null'
InputData['IrregularSampledData'] = {'irr':templat_neo['irr']}
InputData['name'] = 'Motion capture'
for i,sorter in enumerate(MongoReader.block_br.segments):
    if sorter.name!='Motion':
        continue
    motion_capture=MongoReader.block_br.segments[i].irregularlysampledsignals
    motion_capture_array=[]
    for j in motion_capture:
        motion_capture_array.append(np.array(j))
    motion_unit=motion_capture[0].units
    motion_capture_array=np.squeeze(np.array(motion_capture_array))*motion_unit
    InputData['IrregularSampledData']['irr']['signal'] = \
                                            motion_capture_array.transpose()
    InputData['IrregularSampledData']['irr']['times'] = \
                                            motion_capture[0].times
    InputData['IrregularSampledData']['irr']['description'] = \
                                            {'column_label':['x','y','z']}
    break
InputList.append(InputData)



#%% convert AnalogData:EMG
InputData = copy.deepcopy(Template)
InputData['Event'] = 'null'
InputData['IrregularSampledData'] = 'null'
InputData['name'] = 'EMG'
for i,sorter in enumerate(MongoReader.block_br.segments):
    if sorter.name!='EMG':
        continue
    EMG=MongoReader.block_br.segments[i].analogsignals
    for j in EMG:
        InputData['AnalogData'] = {}
        InputData['AnalogData']['ana'] = templat_neo['ana']
        InputData['AnalogData']['ana']['signal'] = j.annotations['NeoData']
        InputData['AnalogData']['ana']['t_start'] = 0*pq.s
        InputData['AnalogData']['ana']['sampling_rate']=j.sampling_rate
        InputData['AnalogData']['ana']['description'] = \
        {'column_label':j.name}
    break
InputList.append(InputData)

#%% operate the dicts
continuous_bhv_block = As.data_input(user_data = InputList,index = 3)

#%% save to appointed path
if args.output is not None:
    joblib.dump(continuous_bhv_block, filename=args.output)
