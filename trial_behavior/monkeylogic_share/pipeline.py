# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:04:22 2022

@author: syj
"""
# bhv data如何从0.4转成mongoneo 0.5

import hdf5storage
import joblib
import argparse
import os
import glob
from user_input_entry import Ml2NeoTrial as mt
FILEPATH = os.path.dirname(os.path.abspath(__file__))
# for test
# raw_dirname="/home/cuilab/Desktop/warmdata/\
# CaeserData/BehaviorData/tdc_caeser20201022002_syjSorter_peeler_zyh"
# %% parse the input arguments
parser = argparse.ArgumentParser(argument_default=None)
parser.add_argument("-f", "--file", type=str, 
                    help="file of behevior data")
parser.add_argument("-o", "--output", type=str, 
                    help="the path to save the results")

args = parser.parse_args()

# #%% convert .mat monkeylogic data
raw_dirname = args.file
bhv_name = glob.glob(os.path.join(raw_dirname,'*.bhv2'))[0].split('/')[-1].split('.')[0]
bhv_mat = [i for i in glob.glob(os.path.join(raw_dirname,'*.mat')) if bhv_name in i][0]
bhvsave = hdf5storage.loadmat(bhv_mat)
bhvkey = [i for i in bhvsave if '_' not in i][0]
bhvsave = bhvsave[bhvkey]

#%% operate the dicts
mlblock = mt.data_input(user_data = bhvsave.squeeze(),index = 2)

#%% save to appointed path
if args.output is not None:
    joblib.dump(mlblock, filename=args.output)
        