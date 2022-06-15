#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:08:11 2021

@author: cuilab
"""
import numpy as np
import quantities as pq
import warnings
import scipy
import pymongo
import scipy.stats
import scipy.signal
from tqdm import tqdm
from .Mongo2Neo import MongoDict2Seg
from neo import Block
from .BaseInterfaceTools import str2pq,GenerateQuery

def spike_time(spiketrains, kargs):
    mag_t_start = kargs['t_start'].rescale(spiketrains[0].units).magnitude
    mag_t_stop = kargs['t_stop'].rescale(spiketrains[0].units).magnitude
    SpikeSlice = [st.times.magnitude[np.where((st.times.magnitude>=mag_t_start) * \
                                              (st.times.magnitude<=mag_t_stop))[0]]-mag_t_start\
                  for st in spiketrains]
 
    return {'SpikeTime':SpikeSlice,'SpikeUnits':str(spiketrains[0].units)}

def time_histogram(spiketrains,SpikeTime, kargsList):
    
    binsize = kargsList[0]['binsize'].rescale(str2pq[SpikeTime[0]['StatisticsSpikeUnits']])
    t_start = kargsList[0]['t_start'].rescale(binsize.units)
    t_stop = kargsList[0]['t_stop'].rescale(binsize.units)
    t_duration = float((t_stop-t_start).magnitude)
    FloatBin = float(binsize.magnitude)
    binnum = np.round(t_duration/FloatBin,10)
        
    SpikeTimeList = []
    for tri in SpikeTime:
        SpikeTrialList = []
        for cell in tri['StatisticsSpike']:
            cellspike = np.frombuffer(cell).astype(np.float32)
            SpikeTrialList.append(cellspike)
        SpikeTimeList.append(SpikeTrialList)
    
    TimeHistogram = []
    for st in SpikeTimeList:
        TimeHistogram.append(np.array([np.histogram(i,bins = int(binnum),
                                       range = (0,t_duration))[0].astype(float) for i in st]).transpose())
    
    return TimeHistogram

def instantaneous_rate(time_array_list,kargsList,cutoff=3.0):
    Bin_list = [np.frombuffer(i['StatisticsSpike']).reshape(i['StatisticsSpikeShape']).transpose() for i in time_array_list]
    sampling_period = kargsList[0]['sampling_period']
    kernel = kargsList[0]['kernel']
    #kargsList[0]['binsize'] = kargsList[0]['sampling_period']
    
    # main function:
    units = pq.CompoundUnit(
        "{}*s".format(sampling_period.rescale('s').item()))   

        
    if cutoff < kernel.min_cutoff:
        cutoff = kernel.min_cutoff
        warnings.warn("The width of the kernel was adjusted to a minimally "
                      "allowed width.")
        
    t_arr = np.arange(-cutoff * kernel.sigma.rescale(units).magnitude,
                      cutoff * kernel.sigma.rescale(units).magnitude +
                      sampling_period.rescale(units).magnitude,
                      sampling_period.rescale(units).magnitude) * units
    
    InstantaneousRate = []
    for time_array in tqdm(Bin_list):
        time_vector_list = list(np.concatenate((time_array, np.zeros((time_array.shape[0],1))),1))
    
        r_list = np.array([scipy.signal.fftconvolve(time_vector,kernel(t_arr).rescale(pq.Hz).magnitude,'same')[0:-1] \
                           for time_vector in time_vector_list]).transpose()
        InstantaneousRate.append(r_list)
    
    return InstantaneousRate

def elephant_preprocessing(i,mldata,kargs,br_gain):
    kargs = kargs.copy()
    mlTrial = {}
    
    '''
    mlTrial['Trial'] = mldata['Trial'][i]
    mlTrial['Block'] = mldata['Block'][i]
    mlTrial['Condition'] = mldata['Condition'][i]
    mlTrial['AbsoluteTrialStartTime'] = mldata['AbsoluteTrialStartTime'][i]
    mlTrial['TrialError'] = mldata['TrialError'][i]
    mlTrial['BehavioralCodes'] = {}
    mlTrial['BehavioralCodes']['CodeNumbers'] = mldata['BehavioralCodes']['CodeNumbers'][i]
    mlTrial['BehavioralCodes']['CodeTimes'] = mldata['BehavioralCodes']['CodeTimes'][i]
    mlTrial['ObjectStatusRecord'] = {}
    mlTrial['ObjectStatusRecord']['Time'] = mldata['ObjectStatusRecord']['Time'][i]
    mlTrial['ObjectStatusRecord']['Status'] = mldata['ObjectStatusRecord']['Status'][i]
    mlTrial['ObjectStatusRecord']['Position'] = mldata['ObjectStatusRecord']['Position'][i]
    mlTrial['UserVars'] = {}
    
    for Userkey in mldata['UserVars'][i]:
        mlTrial['UserVars'][Userkey] = mldata['UserVars'][i][Userkey]'''
        
    for mlKey in mldata.keys():
        
        if isinstance(mldata[mlKey],dict):
            mlTrial[mlKey] = {}
            for subMlKey in mldata[mlKey].keys():
                mlTrial[mlKey][subMlKey] = mldata[mlKey][subMlKey][i]
            continue
        
        if isinstance(mldata[mlKey],list):
            if isinstance(mldata[mlKey][i],dict):
                for subMlKey in mldata[mlKey][i].keys():
                    mlTrial[mlKey][subMlKey] = mldata[mlKey][i][subMlKey]
                continue
        
        mlTrial[mlKey] = mldata[mlKey][i]
    
    if 'aligned_marker2' not in kargs.keys():
        kargs['aligned_marker2'] = kargs['aligned_marker']
        
    CodeIndex = np.array(mlTrial['BehavioralCodes']['CodeNumbers'])==kargs['aligned_marker']
    CodeIndex2 = np.array(mlTrial['BehavioralCodes']['CodeNumbers'])==kargs['aligned_marker2']
    if (mlTrial['TrialError']==kargs['TrialError']) and (sum(CodeIndex)!=0) and (sum(CodeIndex2)!=0):       
        CodeTime = np.array(mlTrial['BehavioralCodes']['CodeTimes'])[CodeIndex]*pq.ms
        CodeTime2 = np.array(mlTrial['BehavioralCodes']['CodeTimes'])[CodeIndex2]*pq.ms
        t_start = CodeTime+mlTrial['AbsoluteTrialStartTime']*pq.ms+br_gain+kargs['t_left']
        t_stop = CodeTime2+mlTrial['AbsoluteTrialStartTime']*pq.ms+br_gain+kargs['t_right']
        mlTrial['t_start'] = {}
        mlTrial['t_start']['magnitude'] = t_start.magnitude[0]
        mlTrial['t_start']['units'] = str(t_start.units)
        mlTrial['t_stop'] = {}
        mlTrial['t_stop']['magnitude'] = t_stop.magnitude[0]
        mlTrial['t_stop']['units'] = str(t_stop.units)
        kargs.update({'t_start':t_start,'t_stop':t_stop})        
        return kargs,mlTrial
        
    '''
    CodeIndex = np.array(mlTrial['BehavioralCodes']['CodeNumbers'])==kargs['aligned_marker']
    if (mlTrial['TrialError']==kargs['TrialError']) and (sum(CodeIndex)!=0):       
        CodeTime = np.array(mlTrial['BehavioralCodes']['CodeTimes'])[CodeIndex]*pq.ms
        TrialCodeTime = CodeTime+mlTrial['AbsoluteTrialStartTime']*pq.ms+br_gain
        t_start = kargs['t_left']+TrialCodeTime
        t_stop = kargs['t_right']+TrialCodeTime
        mlTrial['t_start'] = {}
        mlTrial['t_start']['magnitude'] = t_start.magnitude[0]
        mlTrial['t_start']['units'] = str(t_start.units)
        mlTrial['t_stop'] = {}
        mlTrial['t_stop']['magnitude'] = t_stop.magnitude[0]
        mlTrial['t_stop']['units'] = str(t_stop.units)
        kargs.update({'t_start':t_start,'t_stop':t_stop})        
        return kargs,mlTrial'''

def GetIrrSeg(mycol):
    IrrSeg = [i for i in mycol.find({'Data':'IrregularData'})]
    
class DataPreprocessor():
    def __init__(self,
                 collection,
                 SegName,
                 Saverip="mongodb://localhost:27017/",
                 db="Caesar_monkey",
                 username='admin',
                 password='cuilab324'):
        
        self.myclient = pymongo.MongoClient(Saverip, username=username, password=password)
        self.mydb = self.myclient[db]
        self.mycol = self.mydb[collection]
        self.mldata = {}
        self.mldata = self.mycol.find_one({'name':'BehaviorData'})['mldata']
        self.username = username
        
        BlockDescription = self.mycol.find_one({'name':'Gain'})
        BlockDescription['ml_br_gain'] = BlockDescription['ml_br_gain']*str2pq[BlockDescription['ml_br_gain_unit']]
        BlockDescription['event_start'] = BlockDescription['event_start']*str2pq[BlockDescription['event_start_unit']]
        BlockDescription['ml_start'] = BlockDescription['ml_start']*str2pq[BlockDescription['ml_start_unit']]

        self.block_br = Block(name = collection,
                              description = BlockDescription)
        Seg = MongoDict2Seg(self.mycol,SegName)
        self.block_br.segments.append(Seg)
        self.collection = collection
        self.SegName = SegName
        self.ElectrodeMap = {}
        
    def SpikeStatistics(self,**kargs):
        self.kargs = kargs
        self.InsertDict = {}
        GenerateQuery(self.InsertDict,kargs)
        self.InsertDict['collection'] = self.collection
        self.InsertDict['SegName'] = self.SegName
        self.InsertDict['name'] = 'Statistics'
        
        if 'StatisticsML' in kargs.keys():
            del self.InsertDict['StatisticsML']
            
        StatisticsResults = self._BehaviorStatisticList(self.InsertDict)
        if len(StatisticsResults)!=0: return StatisticsResults
        
        Statistics_mode = kargs['Statistics']        
        spiketrains = self.block_br.segments[0].spiketrains
        
        self.seg_size = kargs['t_right']-kargs['t_left']     
        br_gain = self.block_br.description['ml_br_gain']        

        StatisticsFile = [elephant_preprocessing(i,self.mldata,kargs,br_gain) for i in range(len(self.mldata['Trial']))]
        self.StatisticsMList = [x[1] for x in StatisticsFile if x != None]
        self.kargsList = [x[0] for x in StatisticsFile if x != None]

        GroupName = [sts.description['ElectrodeLabel']+' clu'+ str(int(sts.description['clu'])) for sts in self.block_br.segments[0].spiketrains]
        self.InsertDict['ElectrodeMap'] = GroupName
        self.InsertDict['username'] = self.username
        
        if Statistics_mode == 'time_histogram':
            StatisticsResults = self.GetTimeHistogram(spiketrains,self.kargsList)
        if Statistics_mode == 'instantaneous_rate':
            StatisticsResults = self.GetInstantaneousRate(spiketrains,self.kargsList)
        if Statistics_mode == 'spike_time':
            StatisticsResults = self.GetSpikeTime(spiketrains,self.kargsList)
        
        return StatisticsResults
    
    def GetSpikeTime(self,spiketrains,kargsList):
        
        InsertDict = self.InsertDict.copy()
        InsertDict['Statistics'] = 'spike_time'
        if 'binsize' in InsertDict.keys(): del InsertDict['binsize']
        if 'sampling_period' in InsertDict.keys(): del InsertDict['sampling_period']
        if 'kernel' in InsertDict.keys(): del InsertDict['kernel']
        
        StatisticsResults = self._BehaviorStatisticList(InsertDict)
        if len(StatisticsResults)!=0: return StatisticsResults
        
        SpikeTime = [spike_time(spiketrains,i) for i in tqdm(kargsList)]
        self._InsertFun(InsertDict,SpikeTime)
                
        return self._BehaviorStatisticList(InsertDict)                
    
    def GetTimeHistogram(self,spiketrains,kargsList):
        
        InsertDict = self.InsertDict.copy()
        InsertDict['Statistics'] = 'time_histogram'
        if 'sampling_period' in InsertDict.keys(): del InsertDict['sampling_period']
        if 'kernel' in InsertDict.keys(): del InsertDict['kernel']
        if 'binsize' not in InsertDict.keys(): InsertDict['binsize'] = self.InsertDict['sampling_period']
        
        StatisticsResults = self._BehaviorStatisticList(InsertDict)
        if len(StatisticsResults)!=0: return StatisticsResults
        SpikeTime = self.GetSpikeTime(spiketrains,kargsList)
        TimeHistogram = time_histogram(spiketrains,SpikeTime, kargsList)
        self._InsertFun(InsertDict,TimeHistogram)
                
        return self._BehaviorStatisticList(InsertDict)
    
    def GetInstantaneousRate(self,spiketrains,kargsList):
        
        InsertDict = self.InsertDict.copy()
        InsertDict['Statistics'] = 'instantaneous_rate'
        if 'binsize' in InsertDict.keys(): del InsertDict['binsize']
        
        StatisticsResults = self._BehaviorStatisticList(InsertDict)
        if len(StatisticsResults)!=0: return StatisticsResults
        
        kargsList[0]['binsize'] = self.kargs['sampling_period']
        TimeHistogram = self.GetTimeHistogram(spiketrains,kargsList)
        InstantaneousRate = instantaneous_rate(TimeHistogram,kargsList,cutoff=3.0)
        self._InsertFun(InsertDict,InstantaneousRate)
                
        return self._BehaviorStatisticList(InsertDict)
    
    def DeleteQuery(self,**kargs):
        self.mycol.delete_many(kargs)
    
    def _BehaviorStatisticList(self,InsertDict):
        
        BehaviorStatisticList = []
        if 'StatisticsML' in self.kargs.keys():
            if self.mycol.find_one(InsertDict)!=None:
                if self.kargs['StatisticsML']!=None:
                    for TrialML in self.kargs['StatisticsML']:
                        InsertDict['StatisticsML'] = TrialML
                        BehaviorStatisticList = BehaviorStatisticList+[self.mycol.find_one(InsertDict)]
                    return BehaviorStatisticList
        return [i for i in self.mycol.find(InsertDict)]
    
    def _InsertFun(self,InsertDict,InputList):
        
        if 'spike_time' in InsertDict['Statistics']:
            InsertList = []
            for n,i in enumerate(InputList):
                InsertDict['StatisticsML'] = self.StatisticsMList[n]
                InsertDict['StatisticsSpike'] = [b.tobytes() for b in i['SpikeTime']]
                InsertDict['StatisticsSpikeUnits'] = i['SpikeUnits']
                InsertDict['ElectrodeMap'] = self.InsertDict['ElectrodeMap']
                #if self.mycol.find_one(InsertDict)==None:
                InsertList.append(InsertDict.copy())
            self.mycol.insert_many(InsertList)
            del InsertDict['StatisticsML']
            del InsertDict['StatisticsSpike']
            del InsertDict['StatisticsSpikeUnits']
            del InsertDict['ElectrodeMap']
        else:
            InsertList = []
            for n,i in enumerate(InputList):
                InsertDict['StatisticsML'] = self.StatisticsMList[n]
                InsertDict['StatisticsSpike'] = i.tobytes()
                InsertDict['StatisticsSpikeDtype'] = str(i.dtype)
                InsertDict['StatisticsSpikeShape'] = i.shape
                InsertDict['ElectrodeMap'] = self.InsertDict['ElectrodeMap']
                #if self.mycol.find_one(InsertDict)==None:
                InsertList.append(InsertDict.copy())
            self.mycol.insert_many(InsertList)
            
            del InsertDict['StatisticsML']
            del InsertDict['StatisticsSpike']
            del InsertDict['StatisticsSpikeDtype']
            del InsertDict['StatisticsSpikeShape']
            del InsertDict['ElectrodeMap']