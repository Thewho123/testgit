#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:24:49 2022

@author: cuilab
"""
import os
import joblib
import subprocess
# Initialing:neural_data=None
error_data=['tdc_caeser202010193T001_syjSorter_peeler_wtw','tdc_caeser202010283TALL001_Frank_peeler_wtw',\
            'tdc_caeser202010173T001_Frank_peeler_wtw','tdc_caeser20201024001_syjSorter_peeler_zyh',\
            'tdc_caeser20201021001_lcySorter_peeler_zyh','tdc_caeser20201023002_Frank_peeler_zyh',\
            'tdc_caeser202010143T001_syjSorter_peeler_wtw','tdc_Caesar202010183T001_Frank_peeler_wtw',\
            'tdc_caeser202010153T001_syjSorter_peeler_wtw','tdc_caeser00029DRALL3T001_Frank_peeler_wtw']
# No Initialing
# error_data=['tdc_caeser202010283TALL001_Frank_peeler_wtw','tdc_caeser20201024001_syjSorter_peeler_zyh'\
# ,'tdc_caeser20201021001_lcySorter_peeler_zyh','tdc_caeserDRall10273T001_syjSorter_peeler_wtw'\
# ,'tdc_caeser20201023002_Frank_peeler_zyh','tdc_caeser202010143T001_syjSorter_peeler_wtw'\
# ,'tdc_Caesar202010183T001_Frank_peeler_wtw','tdc_caeser202010153T001_syjSorter_peeler_wtw']

test_data=error_data[5]

save_dir='/home/cuilab/Desktop/warmdata/CaeserData/BehaviorData/'
FILEPATH='/home/cuilab/Desktop/hotdata/data_integration_pipeline/sun_pipeline/neural_data/br_share'
if os.path.exists(os.path.join(save_dir,test_data, 'neural_data.db')):
        os.remove(os.path.join(save_dir,test_data, 'neural_data.db'))

command = 'python ' + os.path.join(FILEPATH,
                                        'pipeline.py')+' -f '+ os.path.join(save_dir,test_data)\
                                        +' -o '+os.path.join(save_dir,test_data,
                                                              'neural_data.db')
ret = subprocess.run(command,shell=True,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if os.path.exists(os.path.join(save_dir,test_data, 'neural_data.db')):
    with open(os.path.join(save_dir,test_data, 'neural_data.db'),'rb') as f:
        try:
            neural_data = joblib.load(f)
        except(EOFError):
            print(test_data+' has neural_data EOFError!')
else:
    neural_data = None

