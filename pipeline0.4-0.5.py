#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 10:31:40 2022

@author: cuilab
"""
import argparse
import subprocess
import os
import joblib
from tqdm import tqdm
import gc
from MongoNeo.interface_layer.MongoNeoInterface import MongoNeoInterface

#%% parse the input arguments
parser = argparse.ArgumentParser(argument_default=None)
parser.add_argument("-f", "--file", type=str,
                    help="file of one monkey's data")
parser.add_argument("-o", "--output", type=str,
                    help="the path to save the results")
parser.add_argument("--user", type=str,
                    help="username of mongodb")
parser.add_argument("--password", type=str,
                    help="password of user")
parser.add_argument("--ip", type=str,
                    help="mongodb ip address")
parser.add_argument("--db", type=str,
                    help="animal name")
args = parser.parse_args()
monkey_dir=args.file
save_dir=args.output
# Can't be used because of CPU storage limitation.Must one data one time
FILEPATH = os.path.dirname(os.path.abspath(__file__))
for data_dir in tqdm(os.listdir(monkey_dir), desc='Converting:'):
    gc.collect()
    print(data_dir,' begin to convert!')
    raw_dirname =os.path.join(monkey_dir,data_dir)
    #------------------------------------------------------------------------------
    # convert discrete behavior data (trial data)
    #------------------------------------------------------------------------------
    if os.path.exists(os.path.join(save_dir,data_dir, 'discrete_behavior.db')):
        os.remove(os.path.join(save_dir,data_dir, 'discrete_behavior.db'))
    
    command = 'python ' + os.path.join(FILEPATH,'trial_behavior',
                                    'monkeylogic_share','pipeline.py')+\
                                    ' -f '+raw_dirname\
                                    +' -o '+os.path.join(save_dir,data_dir,
                                                  'discrete_behavior.db')
    ret = subprocess.run(command,shell=True,
                      stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    del ret
    gc.collect()
    if os.path.exists(os.path.join(save_dir,data_dir, 'discrete_behavior.db')):
        with open(os.path.join(save_dir,data_dir, 'discrete_behavior.db'),'rb') as f:
            try:
                discrete_behavior = joblib.load(f)
            except(EOFError):
                print(data_dir+' has discrete_behavior EOFError!')
    else:
        discrete_behavior = None
        print(data_dir,' has no behavior data!')
        continue
    #------------------------------------------------------------------------------
    # convert neural data (spiketrain, lfp, and event marker)
    #------------------------------------------------------------------------------
    if os.path.exists(os.path.join(save_dir,data_dir, 'neural_data.db')):
        os.remove(os.path.join(save_dir,data_dir, 'neural_data.db'))
    
    command = 'python ' + os.path.join(FILEPATH,'neural_data','br_share',
                                        'pipeline.py')+' -f '+raw_dirname\
                                        +' -o '+os.path.join(save_dir,data_dir,
                                                              'neural_data.db')
    ret = subprocess.run(command,shell=True,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print('Progress finished with '+str(ret.returncode)+'!')
    if os.path.exists(os.path.join(save_dir,data_dir, 'neural_data.db')):
        with open(os.path.join(save_dir,data_dir, 'neural_data.db'),'rb') as f:
            try:
                neural_data = joblib.load(f)
            except(EOFError):
                print(data_dir+' has neural_data EOFError!')
    else:
        neural_data = None
    # ------------------------------------------------------------------------------
    # convert continuous behavior data (EMG and jrajectory)
    #------------------------------------------------------------------------------
    if os.path.exists(os.path.join(save_dir,data_dir, 'continuous_behavior.db')):
        os.remove(os.path.join(save_dir,data_dir, 'continuous_behavior.db'))
    
    command = 'python ' + os.path.join(FILEPATH,'continuous_behavior','aie_share',
                                        'pipeline.py')+' -f '+raw_dirname\
                                        +' -o '+os.path.join(save_dir,data_dir,
                                                        'continuous_behavior.db')
    ret = subprocess.run(command,shell=True,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if os.path.exists(os.path.join(save_dir,data_dir, 'continuous_behavior.db')):
        with open(os.path.join(save_dir,data_dir, 'continuous_behavior.db'),'rb') as f:
            try:
                continuous_behavior = joblib.load(f)
            except(EOFError):
                print(data_dir+' has continuous_behavior EOFError!')
    else:
        continuous_behavior = None
    # %% upload to mongodb
    if args.user is not None:
        mni = MongoNeoInterface(collection = data_dir,dbName=args.db,
                                MongoAddress=args.ip, username=args.user,
                                password=args.password)
        if discrete_behavior is not None:
            mni.Sent2Mongo(discrete_behavior)
            discrete_behavior=None
        if neural_data is not None:
            mni.Sent2Mongo(neural_data)
            neural_data=None
            gc.collect()
        if continuous_behavior is not None:
            mni.Sent2Mongo(continuous_behavior)
            continuous_behavio=None

    