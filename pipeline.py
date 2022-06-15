#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 22:03:01 2021

@author: cuilab
"""
import argparse
import subprocess
import os
import joblib
from MongoNeo.interface_layer.MongoNeoInterface import MongoNeoInterface

#%% parse the input arguments
parser = argparse.ArgumentParser(argument_default=None)
parser.add_argument("-f", "--file", type=str,
                    help="file of experiment data")
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
parser.add_argument("-c","--collection", type=str,
                    help="experiment name")

args = parser.parse_args()
raw_dirname = os.path.abspath(args.file)

#%% call pipeline of three type of neo data
FILEPATH = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------------
# convert discrete behavior data (trial data)
#------------------------------------------------------------------------------

command = 'python ' + os.path.join(FILEPATH,'discrete_behavior',
                                   'monkeylogic_share','pipeline.py')+\
                                    ' -f '+raw_dirname\
                                    +' -o '+os.path.join(args.output,
                                                 'discrete_behavior.db')

ret = subprocess.run(command,shell=True,
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE)

if os.path.exists(os.path.join(args.output, 'discrete_behavior.db')):
    with open(os.path.join(args.output, 'discrete_behavior.db'),'rb') as f:
        discrete_behavior = joblib.load(f)
else:
    discrete_behavior = None

#------------------------------------------------------------------------------
# convert neural data (spiketrain, lfp, and event marker)
#------------------------------------------------------------------------------

command = 'python ' + os.path.join(FILEPATH,'neural_data','br_share',
                                   'pipeline.py')+' -f '+raw_dirname\
                                    +' -o '+os.path.join(args.output,
                                                         'neural_data.db')
ret = subprocess.run(command,shell=True,
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE)

if os.path.exists(os.path.join(args.output, 'neural_data.db')):
    with open(os.path.join(args.output, 'neural_data.db'),'rb') as f:
        neural_data = joblib.load(f)
else:
    neural_data = None

#------------------------------------------------------------------------------
# convert continuous behavior data (EMG and jrajectory)
#------------------------------------------------------------------------------

command = 'python ' + os.path.join(FILEPATH,'continuous_behavior','aie_share',
                                   'pipeline.py')+' -f '+raw_dirname\
                                    +' -o '+os.path.join(args.output,
                                                    'continuous_behavior.db')

ret = subprocess.run(command,shell=True,
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE)

if os.path.exists(os.path.join(args.output, 'continuous_behavior.db')):
    with open(os.path.join(args.output, 'continuous_behavior.db'),'rb') as f:
        continuous_behavior = joblib.load(f)
else:
    continuous_behavior = None

#%% upload to mongodb
if args.user is not None:
    mni = MongoNeoInterface(collection = args.collection,dbName=args.db,
                            MongoAddress=args.ip, username=args.user,
                            password=args.password)

    if discrete_behavior is not None:
        mni.Sent2Mongo(discrete_behavior)

    if neural_data is not None:
        mni.Sent2Mongo(neural_data)

    if continuous_behavior is not None:
        mni.Sent2Mongo(continuous_behavior)
