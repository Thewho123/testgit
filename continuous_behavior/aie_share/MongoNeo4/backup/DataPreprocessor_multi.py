#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:00:19 2020

@author: cuilab
"""


from elephant.statistics import mean_firing_rate
import numpy as np
import quantities as pq
import random
import neo
import warnings
import scipy.stats
import scipy.signal
from tqdm import tqdm
from joblib import Parallel,delayed
from .Mongo2Neo import MongoDict2Seg
import pymongo
from neo import Block

def GenerateQuery(InsertDict,kargs):
    for DictName in kargs.keys():
        if DictName=='StatisticsML':
            InsertDict[DictName] = kargs[DictName]
            continue
        try:
            Magnitude = float(kargs[DictName].magnitude)
            unit = kargs[DictName].units
            InsertDict[DictName] = {}
            InsertDict[DictName]['magnitude'] = Magnitude
            InsertDict[DictName]['unit'] = str(unit)
            continue
        except:
            pass
        try:
            InsertDict[DictName] = float(kargs[DictName])
            continue
        except:
            pass
        
        try:
            InsertDict[DictName] = str(kargs[DictName])
            continue
        except:
            pass
        InsertDict[DictName] = kargs[DictName]

def psth_bootstrap(dataset,kargs):
    
    bootstrap_dataset = []
    for i in range(kargs['rep_num']):
        rand_index = [random.randint(0, dataset.shape[0]-1) for i in range(dataset.shape[0])]
        bootstrap_dataset.append(np.squeeze(np.mean(dataset[rand_index,:,:],0)))
    return np.array(bootstrap_dataset)

def std_psth(dataset,kargs):
    return np.array(dataset)


def get_mean_firing(spiketrains, kargs):
    sts_cut = [mean_firing_rate(st,t_start=kargs['t_start'], t_stop=kargs['t_stop']) for st in spiketrains]
    return np.array(sts_cut)
    
def time_histogram(spiketrains, kargs):
    
    binsize = kargs['binsize']
    t_start = kargs['t_start'].rescale(binsize.units)
    t_stop = kargs['t_stop'].rescale(binsize.units)
    binnum = (t_stop-t_start)/binsize
    
    bs = []
    for st in spiketrains:
        stinput = st.time_slice(t_start=t_start, t_stop=t_stop)
        
        if len(stinput)>2:
            rescale_t_start = t_start.rescale(stinput.units)
            rescale_t_stop = t_stop.rescale(stinput.units)
            Tbs = np.histogram(stinput.magnitude,
                               bins = int(binnum),
                               range = (float(rescale_t_start.magnitude),
                                        float(rescale_t_stop.magnitude)))[0][np.newaxis,:]
            if Tbs.max()/kargs['binsize'].rescale(1*pq.s).magnitude>300:
                if kargs['binsize']>5*pq.ms:
                    return
                elif Tbs.max()>3:
                    return
                    
            #Tbs = conv.BinnedSpikeTrain(stinput, t_start=t_start, t_stop=t_stop,binsize=binsize).to_sparse_array().toarray()
        else:
            Tbs = np.zeros((1,int(binnum)),dtype = np.int64)
        bs.append(Tbs)
    bin_hist = np.array(bs).squeeze()* pq.dimensionless
    
    del spiketrains
    return neo.AnalogSignal(signal=bin_hist.transpose(),
                            sampling_period=binsize, units=bin_hist.units,
                            t_start=t_start)


def instantaneous_rate(spiketrains,kargs,cutoff=3.0):
    
    sampling_period = kargs['sampling_period']
    kargs['binsize'] = kargs['sampling_period']
    kernel = kargs['kernel']
    t_start = kargs['t_start']
    t_stop = kargs['t_stop']
    
    # main function:
    units = pq.CompoundUnit(
        "{}*s".format(sampling_period.rescale('s').item()))
    spiketrains = [spiketrain.rescale(units) for spiketrain in spiketrains]
    
    
    t_start = t_start.rescale(spiketrains[0].units)
    t_stop = t_stop.rescale(spiketrains[0].units)

        
    if cutoff < kernel.min_cutoff:
        cutoff = kernel.min_cutoff
        warnings.warn("The width of the kernel was adjusted to a minimally "
                      "allowed width.")
        
    t_arr = np.arange(-cutoff * kernel.sigma.rescale(units).magnitude,
                      cutoff * kernel.sigma.rescale(units).magnitude +
                      sampling_period.rescale(units).magnitude,
                      sampling_period.rescale(units).magnitude) * units
    
    time_array = time_histogram(spiketrains, kargs)
    
    try:
        if time_array==None: return
    except:
        pass
    
    time_array = time_array.transpose()
    
    time_vector_list = list(np.concatenate((time_array, np.zeros((time_array.shape[0],1))),1))
    
    r_list = [scipy.signal.fftconvolve(time_vector,kernel(t_arr).rescale(pq.Hz).magnitude,'full') \
              for time_vector in time_vector_list]
    
    r_list = [r[kernel.median_index(t_arr):-(kernel(t_arr).size -
                                       kernel.median_index(t_arr))]\
                                      for r in r_list]

    rate = neo.AnalogSignal(signal=np.array(r_list).transpose(),
                            sampling_period=sampling_period,
                            units=pq.Hz, t_start=t_start, t_stop=t_stop)
    del spiketrains
    return rate

def spike_time(spiketrains, kargs):
    
    sts_cut = []
    for st in spiketrains:
        st_time = st.time_slice(t_start=kargs['t_start'], t_stop=kargs['t_stop'])
        
        if len(st_time) > 0:
            st_time = st_time-kargs['t_start']
            
        else:
            st_time.t_start = kargs['t_start']-kargs['t_start']
            st_time.t_stop = kargs['t_stop']-kargs['t_start']
            
        sts_cut.append(st_time)
   
    del spiketrains
    return sts_cut

def elephant_preprocessing(i,mldata,kargs,br_gain):
    kargs = kargs.copy()
    mlTrial = {}
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
        return kargs,mlTrial

str2pq = {'1.0 s':pq.s,
          '1.0 ms':pq.ms}

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
        #self.InsertDict = self.kargs.copy()
        
        '''
        for key in self.InsertDict.keys():
            try:
                magnitude = float(self.kargs[key].magnitude)
                Units = str(self.kargs[key].units)
                self.InsertDict[key] = {}
                self.InsertDict[key]['magnitude'] = magnitude
                self.InsertDict[key]['Units'] = Units
            except:
                pass
        if 'kernel' in self.InsertDict.keys(): self.InsertDict['kernel'] = str(self.kargs['kernel'])'''
        self.InsertDict = {}
        GenerateQuery(self.InsertDict,kargs)
        self.InsertDict['collection'] = self.collection
        self.InsertDict['SegName'] = self.SegName
        self.InsertDict['name'] = 'Statistics'
        par = self.InsertDict.copy()
        
        if 'StatisticsML' in kargs.keys():
            del self.InsertDict['StatisticsML']
            if kargs['StatisticsML']!=None:
                BehaviorStatisticList = []
                if self.mycol.find_one(self.InsertDict)!=None:
                    for TrialML in kargs['StatisticsML']:
                        self.InsertDict['StatisticsML'] = TrialML
                        BehaviorStatisticList = BehaviorStatisticList+[self.mycol.find_one(self.InsertDict)]
                    return BehaviorStatisticList
                
        if 'StatisticsML' in self.InsertDict.keys():
            if self.InsertDict['StatisticsML'] == None:
                del self.InsertDict['StatisticsML']
                    
        if self.mycol.find_one(self.InsertDict)!=None: return [i for i in self.mycol.find(self.InsertDict)]
        
        SpikeStatisticsFile = {}
        SpikeStatisticsFile.update(kargs)
        Statistics_mode = kargs['Statistics']        
        #return
        spiketrains = self.block_br.segments[0].spiketrains
        #spiketrains = [SpikeTrain(times = np.array(i)*pq.s,t_stop = np.array(i)[-1]*pq.s) for i in spiketrains]
        
        self.seg_size = kargs['t_right']-kargs['t_left']
        
        br_gain = self.block_br.description['ml_br_gain']        

        StatisticsFile = [elephant_preprocessing(i,self.mldata,kargs,br_gain) for i in range(len(self.mldata['Trial']))]
        StatisticsMList = [x[1] for x in StatisticsFile if x != None]
        kargsList = [x[0] for x in StatisticsFile if x != None]
        self.kargsList = kargsList
        StatisticsSpikeList = []
        if Statistics_mode == 'time_histogram':
            Seg = set(range(0,len(kargsList),300))
            Seg.add(len(kargsList))
            Seg = np.sort(list(Seg))
            for j in range(len(Seg)-1):
                StatisticsSpikeList = StatisticsSpikeList+Parallel(n_jobs=-1)(delayed(time_histogram)(spiketrains,i)\
                                                                              for i in tqdm(kargsList[Seg[j]:Seg[j+1]]))
            #for i in tqdm(kargsList): a = time_histogram(spiketrains,i)
        if Statistics_mode == 'instantaneous_rate':
            Seg = set(range(0,len(kargsList),300))
            Seg.add(len(kargsList))
            Seg = np.sort(list(Seg))
            for j in range(len(Seg)-1):
                StatisticsSpikeList = StatisticsSpikeList+Parallel(n_jobs=-1)(delayed(instantaneous_rate)(spiketrains,i)\
                                                                              for i in tqdm(kargsList[Seg[j]:Seg[j+1]]))
            #for i in tqdm(kargsList): a = instantaneous_rate(spiketrains,i)
        if Statistics_mode == 'spike_time':
            Seg = set(range(0,len(kargsList),300))
            Seg.add(len(kargsList))
            Seg = np.sort(list(Seg))
            for j in range(len(Seg)-1):
                StatisticsSpikeList = StatisticsSpikeList+Parallel(n_jobs=-1)(delayed(spike_time)(spiketrains,i)\
                                                                              for i in tqdm(kargsList[Seg[j]:Seg[j+1]]))
           #for i in tqdm(kargsList): a = spike_time(spiketrains,i)
        
        SpikeListTemp = []
        MListTemp = []
        for n, x in enumerate(StatisticsMList):
            try:
                if StatisticsSpikeList[n] != None:
                    SpikeListTemp.append(StatisticsSpikeList[n])
                    MListTemp.append(x)
            except:
                SpikeListTemp.append(StatisticsSpikeList[n])
                MListTemp.append(x)
            
        StatisticsMList = MListTemp
        StatisticsSpikeList = SpikeListTemp
        
        SpikeStatisticsFile.update({'StatisticsSpikeList':StatisticsSpikeList,\
                                    'StatisticsMList':StatisticsMList})
        try:    
            spike_diff_shape = np.where(np.diff([i.flatten().shape[0] for i in SpikeStatisticsFile['StatisticsSpikeList']])!=0)[0]
            spike_del = spike_diff_shape[np.where(np.diff(spike_diff_shape))[0]]+1
            for i in spike_del:
                del SpikeStatisticsFile['StatisticsSpikeList'][i]
                del SpikeStatisticsFile['StatisticsMList'][i]
        except:
            pass
        
        GroupName = [sts.description['ElectrodeLabel']+' clu'+ str(int(sts.description['clu'])) for sts in self.block_br.segments[0].spiketrains]
        self.InsertDict['ElectrodeMap'] = GroupName
        self.InsertDict['username'] = self.username
        
        if Statistics_mode == 'time_histogram':
            for n,i in enumerate(SpikeStatisticsFile['StatisticsSpikeList']):
                self.InsertDict['StatisticsML'] = SpikeStatisticsFile['StatisticsMList'][n]
                self.InsertDict['StatisticsSpike'] = i.tobytes()
                self.InsertDict['StatisticsSpikeDtype'] = str(i.dtype)
                self.InsertDict['StatisticsSpikeShape'] = i.shape
                if self.mycol.find_one(self.InsertDict)==None:
                    InsertResult = self.mycol.insert_one(self.InsertDict.copy())
        #InserList = [] 
        if Statistics_mode == 'instantaneous_rate':
            for n,i in enumerate(SpikeStatisticsFile['StatisticsSpikeList']):
                self.InsertDict['StatisticsML'] = SpikeStatisticsFile['StatisticsMList'][n]
                self.InsertDict['StatisticsSpike'] = i.tobytes()
                self.InsertDict['StatisticsSpikeDtype'] = str(i.dtype)
                self.InsertDict['StatisticsSpikeShape'] = i.shape
                #InserList.append(self.InsertDict.copy())
                if self.mycol.find_one(self.InsertDict)==None:
                    InsertResult = self.mycol.insert_one(self.InsertDict.copy())
        #return InserList
        if Statistics_mode == 'spike_time':
            for n,i in enumerate(SpikeStatisticsFile['StatisticsSpikeList']):
                self.InsertDict['StatisticsML'] = SpikeStatisticsFile['StatisticsMList'][n]
                self.InsertDict['StatisticsSpike'] = [b.times.magnitude.tobytes() for b in i]
                self.InsertDict['StatisticsSpikeUnits'] = str(i[0].units)
                if self.mycol.find_one(self.InsertDict)==None:
                    InsertResult = self.mycol.insert_one(self.InsertDict.copy())
                    
        self.InsertDict = par
        
        if 'StatisticsML' in kargs.keys():
            del self.InsertDict['StatisticsML']
            if self.mycol.find_one(self.InsertDict)!=None:
                BehaviorStatisticList = []
                if kargs['StatisticsML']!=None:
                    for TrialML in kargs['StatisticsML']:
                        self.InsertDict['StatisticsML'] = TrialML
                        BehaviorStatisticList = BehaviorStatisticList+[self.mycol.find_one(self.InsertDict)]
                    return BehaviorStatisticList
    
                
        return [i for i in self.mycol.find(self.InsertDict)]
    
    def getTrialTime(self,**kargs):
        br_gain = -self.block_br.description['ml_start']
        StatisticsFile = [elephant_preprocessing(i,kargs,br_gain) for i in self.mldata]
        kargsList = [x[0] for x in StatisticsFile if x != None]
        return kargsList
        
    def psth(self,SpikeStatisticsFile,**kargs):
    
        trial_index = kargs['trial_index']
        Statistics_mode = kargs['psth_Statistics']
        
        PSTH_Statistics = {'bootstrap': psth_bootstrap, 
                          'None': std_psth}
        
        dataset = []
        
        for i in trial_index:
            trial_data = SpikeStatisticsFile['StatisticsSpikeList'][i].transpose()
            
            if len(trial_data.shape)<2:
                continue
            dataset.append(trial_data)
        dataset = np.array(dataset)
        
        firing_rate = PSTH_Statistics.get(Statistics_mode)(dataset,kargs) 
        
        psth = {'Mean':np.mean(firing_rate,0),
                'std':np.std(firing_rate,0),
                #'data':firing_rate,
                'sample_method':Statistics_mode}
    
        return psth    
    def DeleteQuery(self,**kargs):
        self.mycol.delete_many(kargs)
'''
    def affinewarp_method(self,t_left = -1000*pq.ms,t_right = 1000*pq.ms,
                          aligned_marker = 5,TrialError = 0,binsize = 1*pq.ms):
        
        time_hist = self.NeuralData.SpikeStatistics(Statistics = 'time_histogram',
                                                           binsize = binsize,
                                                            t_left = t_left, 
                                                     tdc       t_right = t_right,
                                                            aligned_marker = aligned_marker,
                                                            TrialError = TrialError)
        
        
        
        
        group_index = self.NeuralDataFile['time_histogram']['group_index']
        trial_index = group_index['trial_index']
        group_index.update({'All_trial':np.ones((trial_index.shape[0],))==1})
        del group_index['trial_index']
        
        warping = {}
        pbar = tqdm(group_index.items())
        pbar.set_description("Processing: %s" % 'time_warping')
        
        for key,value in pbar:
            
            binned = np.array(time_hist['StatisticsSpikeList']).astype(int)[trial_index['value'],:,:]
            align_sniff = PiecewiseWarping(n_knots=1, warp_reg_scale=1e-6, smoothness_reg_scale=20.0)
            align_sniff.manual_fit(binned, iterations=50, warp_iterations=200)
            warping[key] = {}
            warping[key]['raw_data'] = binned
            warping[key]['transformed_data'] = align_sniff.transform(binned)
            
        return warping'''
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
