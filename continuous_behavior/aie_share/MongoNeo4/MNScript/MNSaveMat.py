#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 18:47:20 2021

@author: cuilab
"""
import os
import scipy.io as scio
import quantities as pq
from elephant.kernels import GaussianKernel
import pymongo
from MongoNeo.MongoReader import MongoReadModule
#%% save mat file

raw_dir = 'D:\\postgraduate\\ArrayDataAnalyse\\'
peeler_dirlist = pymongo.MongoClient("mongodb://10.10.47.78:27017/", username='yongxiang', 
                                     password='cuilab322')
['Hilbert_monkey_threads'].list_collection_names()
peeler_dirlist = [i for i in peeler_dirlist if '.' not in i and '_DP' not in i]
for CollectionName in peeler_dirlist:
    CollectionName= 'tdc_caeser20201021001_lcySorter_peeler_zyh'
    MongoReader = MongoReadModule(collection = CollectionName,
                                  SegName = 'Kilo_spike_train',
                                  Saverip="mongodb://10.10.47.78:27017/",
                                  db='Caesar_monkey_array',
                                  username='yongxiang',
                                  password='cuilab322',
                                  LFP=False)
    #MongoReader.gain = 12.45*pq.s


    instantaneous_rate2 = MongoReader.SpikeStatisticsReader(Statistics = 'instantaneous_rate',
                                                           sampling_period = 50*pq.ms, 
                                                           t_left = -1000*pq.ms, 
                                                           t_right = 1000*pq.ms,
                                                           kernel = GaussianKernel(50*pq.ms),
                                                           aligned_marker = 5,
                                                           TrialError=0,
                                                           DeleteQuery=True)
    
    instantaneous_rate = MongoReader.SpikeStatisticsReader(Statistics = 'time_histogram',
                                                           binsize = 50*pq.ms, 
                                                           t_left = -1000*pq.ms, 
                                                           t_right = 00*pq.ms,
                                                           aligned_marker = 5,
                                                           TrialError=0,
                                                           DeleteQuery=True)
    
    instantaneous_rate = MongoReader.SpikeStatisticsReader(Statistics = 'spike_time',
                                                           t_left = -300*pq.ms, 
                                                           t_right = 500*pq.ms,
                                                           aligned_marker = 5,
                                                           aligned_1st_marker = False,
                                                           TrialError=0,
                                                           DeleteQuery=False)
    
    
    instantaneous_rate = MongoReader.SpikeStatisticsReader(**kargs)
    
    Pos = np.array([i.irregularlysampledsignals[2][-2][2] for i in MongoReader.StatisticsTrialList])
    reg = MultiTaskLassoCV(cv=5).fit(a[0:700], Pos[0:700])
    from sklearn.metrics import r2_score
    r2_score(Pos, reg.predict(a))
    reg = MultiTaskLassoCV(cv=5).fit(a[0:700], Pos[0:700])
    data = np.array(instantaneous_rate)
    b = reg.predict(a[700::])
    c = [np.math.atan2(i[1],i[0])/np.pi*180 for i in b]
    bb = np.array(c)-np.array(d)
    bb = np.abs(np.array(c)-np.array(d))
    bb[bb>180] = np.abs(bb[bb>180]-360)
    d = np.sort(bb)
    
    ContinuousData = MongoReader.GetAllContinous()
    -2,3
    InstantaneousRate = {}
    InstantaneousRate['NeuralData'] = instantaneous_rate
    InstantaneousRate['mldata'] = MongoReader.StatisticsML
    InstantaneousRate['group'] = [i.description['group'] for i in MongoReader.block_br.segments[0].spiketrains]
    InstantaneousRate['SegID'] = [i.description['SegID'] for i in MongoReader.block_br.segments[0].spiketrains]
    
    SpikeTime = MongoReader.SpikeTimeFun(Statistics = 'spike_time',
                                         t_left = -2000*pq.ms, 
                                         t_right = 2000*pq.ms,
                                         aligned_marker = 2,
                                         aligned_marker2 = 4,
                                         TrialError=0,
                                         StatisticsML = None)
    
    SpikeTimeData = {}
    SpikeTimeData['NeuralData'] = SpikeTime
    SpikeTimeData['mldata'] = MongoReader.StatisticsML
    SpikeTimeData['group'] = [i.description['group'] for i in MongoReader.block_br.segments[0].spiketrains]
    SpikeTimeData['SegID'] = [i.description['SegID'] for i in MongoReader.block_br.segments[0].spiketrains]
    
    SpikeBin = MongoReader.SpikeBin(Statistics = 'time_histogram',
                                    binsize = 50*pq.ms, 
                                    t_left = -2000*pq.ms, 
                                    t_right = 2000*pq.ms,
                                    aligned_marker = 2,
                                    aligned_marker2 = 4,
                                    TrialError=0,
                                    StatisticsML = None)
    
    TimeHistogram = {}
    TimeHistogram['NeuralData'] = SpikeBin
    TimeHistogram['mldata'] = MongoReader.StatisticsML
    TimeHistogram['group'] = [i.description['group'] for i in MongoReader.block_br.segments[0].spiketrains]
    TimeHistogram['SegID'] = [i.description['SegID'] for i in MongoReader.block_br.segments[0].spiketrains]
    
    NeuralDataFile = {}
    NeuralDataFile['InstantaneousRate'] = InstantaneousRate
    NeuralDataFile['TimeHistogram'] = TimeHistogram
    NeuralDataFile['SpikeTimeData'] = SpikeTimeData
    
    
    date=CollectionName[CollectionName.find('2'):CollectionName.find('2')+8]
    rawpath1 = os.path.join(raw_dir,date)
    rawpath = os.path.join(rawpath1,'bhv_neural')
    FileName ='NeuralDataFile_aligned_(10ms)'+str(aligned_marker)
    if os.path.exists(os.path.join(rawpath,FileName+'.mat')):
        os.remove(os.path.join(rawpath,FileName+'.mat'))
    scio.savemat(os.path.join(rawpath,FileName+'.mat'),NeuralDataFile)