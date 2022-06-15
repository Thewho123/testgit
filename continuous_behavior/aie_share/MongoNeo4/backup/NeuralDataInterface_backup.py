#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:12:28 2020

@author: cuilab
"""

from tridesclous import DataIO
from neo import io
import scipy.io as scio
import quantities as pq
from neo import Block,Segment,SpikeTrain,Event,IrregularlySampledSignal,AnalogSignal
import os
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
import json
from brpylib import NsxFile
import hdf5storage
from mountainlab_pytools import mdaio
from multiprocessing import Pool
from functools import  partial
import time
from MongoNeo.BehaviorInterface import BehaviorMetaDataInterface
from affinewarp import SpikeData
from tridesclous.iotools import ArrayCollection
import multiprocessing
import shutil
from joblib import Parallel,delayed

try:
    multiprocessing.set_start_method('spawn')
except:
    pass

def ListCompare(Cell,CellList):
    CellSet = set(np.where(Cell.bin_spikes(int(Cell.tmax)).squeeze())[0])
    CellListSet = [set(np.where(i.bin_spikes(int(i.tmax)).squeeze())[0]) for i in CellList]
    interCellListSet = [len(CellSet&i)/len(CellSet) for i in CellListSet]
    return interCellListSet

def SaveMat(i,dataio_dir,matdir):
    arrays = ArrayCollection(parent=None, dirname=os.path.join(dataio_dir,'mda'))
    arrays.load_all()
    RawDataArray = arrays.get('mdaData')
    scio.savemat(os.path.join(matdir,str(i)+'Raw.mat'),{'data':RawDataArray.T[i]})

def TwoSorterCompare(kilosortSeg,tdcSeg):
    mlPool = Pool(20)
    tdcSpikeTimeList = [SpikeData(trials = np.zeros(np.array(i).shape[0]), 
                               spiketimes = np.array(i)*1000, 
                               neurons=np.zeros(np.array(i).shape[0]), 
                               tmin = 0, 
                               tmax = np.ceil(max(np.array(i)*1000))) for i in tqdm(tdcSeg)]
        
    kiloSpikeTimeList = [SpikeData(trials = np.zeros(np.array(i).shape[0]), 
                           spiketimes = np.array(i)*1000, 
                           neurons=np.zeros(np.array(i).shape[0]), 
                           tmin = 0, 
                           tmax = np.ceil(max(np.array(i)*1000))) for i in tqdm(kilosortSeg)]
    
    SpiketrainList = mlPool.map(partial(ListCompare,
                                        CellList=kiloSpikeTimeList),tqdm(tdcSpikeTimeList))
    mlPool.close()
    mlPool.join()
    SpiketrainList = np.array(SpiketrainList)
    return SpiketrainList

def FindEndTrialCodeTime(BehaviorBlock,event_time,event_marker):
    IndexArray = np.array([i.index for i in BehaviorBlock.segments])
    TrialIndex = np.where(IndexArray==max(IndexArray))[0][0]
    MaxTrial = BehaviorBlock.segments[TrialIndex]
    MaxTrialTime = MaxTrial.description['AbsoluteTrialStartTime']+MaxTrial.events[0].times
    Gain = event_time[-1]-MaxTrialTime[np.where(MaxTrial.events[0].labels==event_marker[-1])[0][0]]
    return Gain+MaxTrialTime[-1]

def ReadKiloSort(kilo_dir):
    KiloSortResult = {}
    import pandas as pd
    if len(glob.glob(os.path.join(kilo_dir,'cluster_info*')))!=0:
        info = pd.read_csv(glob.glob(os.path.join(kilo_dir,'cluster_info*'))[0], delimiter='\t')
        KiloSortResult['cluster'] = np.load(glob.glob(os.path.join(kilo_dir,'spike_clusters*'))[0])
        KiloSortResult['time'] = np.load(glob.glob(os.path.join(kilo_dir,'spike_times*'))[0])
        KiloSortResult['id'] = list(info['id'])
        KiloSortResult['group'] = list(info['group'])
        KiloSortResult['ch'] = list(info['ch'])
    else:
        KiloSortResult['cluster'] = np.load(glob.glob(os.path.join(kilo_dir,'spike_clusters*'))[0])
        KiloSortResult['time'] = np.load(glob.glob(os.path.join(kilo_dir,'spike_times*'))[0])
    return KiloSortResult

def Kiloshare(KiloSortResult,nsFile,RecordingParam,sampling_rate=30000*pq.Hz):
    Seg = Segment(name = 'Kilo_spike_train')
    if 'id' in KiloSortResult.keys():
        for ind,i in enumerate(KiloSortResult['id']):
            if not isinstance(KiloSortResult['group'][ind], str):
                continue
            
            if 'good' not in KiloSortResult['group'][ind]:
                continue
            '''
            RawFile = os.path.join(os.path.dirname(nsFile),'mda')
            arrays = ArrayCollection(parent=None, dirname=RawFile)
            arrays.load_all()
            RawDataArray = arrays.get('mdaData')[:,KiloSortResult['ch'][ind]]
            
            waveforms=np.array([RawDataArray[(int(index)-16):(int(index)+32)]\
                                                  for index in KiloSortResult['time'][KiloSortResult['cluster']==i].squeeze()])'''
            spike_description = {'clu':str(i),
                                 'group':str(ind)}

            spike_description.update(RecordingParam[KiloSortResult['ch'][ind]])

            KiloSpike = KiloSortResult['time'][KiloSortResult['cluster']==i].squeeze()/sampling_rate
            if len(KiloSpike)<5000: continue
            train = SpikeTrain(times = KiloSpike,
                               t_stop = KiloSpike[-1],
                               sampling_rate=sampling_rate,
                               description = spike_description)
            Seg.spiketrains.append(train)
    else:
        for i in np.unique(KiloSortResult['cluster']):
            KiloSpike = KiloSortResult['time'][KiloSortResult['cluster']==i]/sampling_rate
            if len(KiloSpike)<5000: continue
            train = SpikeTrain(times = KiloSpike,
                               t_stop = KiloSpike[-1],
                               sampling_rate=sampling_rate)
            Seg.spiketrains.append(train)
    return Seg

def kiloSeg(dirname,RecordingParam,sampling_rate):
    KiloSortResult = ReadKiloSort(dirname)
    nsFile = glob.glob(os.path.join(os.path.dirname(dirname),'*ns6'))[0]
    Seg = Kiloshare(KiloSortResult,nsFile,RecordingParam,sampling_rate)
    return Seg

def SpykingcircusReadFun(dirname,RecordingParam,sampling_rate):
    sc_dir = dirname
    with open(sc_dir,"rb") as f:
        Seg = joblib.load(f)
    return Seg

def Herdingspike2ReadFun(dirname,RecordingParam,sampling_rate):
    hs2_dir = dirname
    with open(hs2_dir,"rb") as f:
        Seg = joblib.load(f)
    return Seg

def set_spike_segment(i,RecordingParam,dataio_dir):
    dataio_br = DataIO(dirname=dataio_dir)
    #%% align segments spike data
    memmap = dataio_br.get_spikes(chan_grp=i,seg_num = 0)
    spike_time = memmap['index'].copy()
    cluster_label = memmap['cluster_label'].copy()
    #print('cluster:',cluster_num)
    cluster_num = list(dataio_br.load_catalogue(chan_grp=i)['cluster_labels'])
    processed_signals = dataio_br.arrays[i][0].get('processed_signals')
    
    #%% set spiketrains list
    TrainList = []
    
    for clu in cluster_num:
        
        spike_index = spike_time[cluster_label==clu]
        if len(spike_index)==0:
            continue
        Select = (spike_index<(len(processed_signals)-32)) * (spike_index>16)
        spike_index = spike_index[Select]

        cluster_spike_times = (spike_index/dataio_br.sample_rate)*pq.s
        waveforms=np.array([processed_signals[(index-16):(index+32)]\
                                               for index in spike_index])
            
        #QuatoIndex = int(processed_signals.shape[0]/4)
        peak_amplitude = processed_signals[spike_index]
        mad = np.median(np.abs(processed_signals-np.median(processed_signals)))  
        snr = np.mean(np.abs(peak_amplitude))/(mad*1.4826) 
        spike_description = {'clu':str(clu),
                             'snr':snr,
                             'group':i,
                             'mean_waveform':waveforms.squeeze().mean(0)}
        
        spike_description.update(RecordingParam[i])
        
        train = SpikeTrain(times = cluster_spike_times,
                           t_stop = cluster_spike_times[-1],
                           sampling_rate=dataio_br.sample_rate*pq.Hz,
                           waveforms=waveforms[:,np.newaxis,:],
                           description=spike_description)
        #print(spike_description)
        
        TrainList.append(train)
    dataio_br.arrays[i][0].detach_array('spikes')
    dataio_br.arrays[i][0].detach_array('processed_signals')
    #del dataio_br
    return TrainList

def tdcReadFun(json_dir,RecordingParam,sampling_rate):
    dataio_dir = os.path.dirname(json_dir)
    raw_file = glob.glob(os.path.join(os.path.join(dataio_dir,'mda'),'*.raw'))[0]
    with open(json_dir,'r',encoding='utf-8')as fp:
            json_data = json.load(fp)
        
    json_data['datasource_kargs']['filenames'] = [raw_file]
    
    os.remove(json_dir)
    with open(json_dir,'w',encoding='utf-8') as json_file:
        json.dump(json_data,json_file)
    dataio_br = DataIO(dirname=dataio_dir)
    
    mlPool = Pool(20)
    dataio_br = DataIO(dirname=dataio_dir)
    Seg = Segment(name = 'tdc_spike_train')
    
    SpiketrainList = mlPool.map(partial(set_spike_segment,
                                        RecordingParam=RecordingParam,
                                        dataio_dir=dataio_dir),tqdm(dataio_br.channel_groups.keys()))
    mlPool.close()
    mlPool.join()
    SpiketrainList = [i for i in SpiketrainList if isinstance(i,list)]
    #for i in tqdm(dataio_br.channel_groups.keys()): set_spike_segment(i,RecordingParam,EndTrialCodeTime,dataio_dir)
    
    for i in SpiketrainList: 
        for j in i: 
            if len(j)>5000:
                Seg.spiketrains.append(j)
    return Seg

def IronClustShare(ic_dir,RecordingParam,sampling_rate):
    Seg = Segment(name = 'IronClust_spike_train')
    icFile = mdaio.readmda(ic_dir).T
    for i in np.unique(icFile[:,0]):
        icChData = icFile[icFile[:,0]==i]
        for j in np.unique(icChData[:,-1]):
            icChCluData = icChData[icChData[:,-1]==j]
            if icChCluData.shape[0]<5000: continue
            spike_description = {'clu':j-1,
                                 'group':i}
            train = SpikeTrain(times = icChCluData[:,1]/sampling_rate,
                               t_stop = icChCluData[-1,1]/sampling_rate,
                               sampling_rate=sampling_rate,
                               description=spike_description)
            Seg.spiketrains.append(train)
    return Seg
        
def HDSortShare(hd_dir,RecordingParam,sampling_rate):
    Seg = Segment(name = 'HDSort_spike_train')
    CellUnits = hdf5storage.loadmat(hd_dir)['Units'].squeeze()
    for n,i in enumerate(CellUnits):
        hdFile = i['detectionChannel']
        for j in np.unique(hdFile):
            hdChFile = i['spikeTrain'][hdFile==j]
            if hdChFile.shape[0]<5000:continue
            spike_description = {'clu':n,
                                 'group':j-1}
            train = SpikeTrain(times = hdChFile/sampling_rate,
                               t_stop = hdChFile[-1]/sampling_rate,
                               sampling_rate=sampling_rate,
                               description=spike_description)
            Seg.spiketrains.append(train)
    return Seg

def BlockUpdate(dirname,Segname,blockdata,RecordingParam,sampling_rate):
    IndexName = [i.name for i in blockdata.segments]
    ShareFunction = {'tdc_spike_train':tdcReadFun,
                     'Kilo_spike_train':kiloSeg,
                     'Spykingcircus_spike_train':SpykingcircusReadFun,
                     'Herdingspike2_spike_train':Herdingspike2ReadFun,
                     'IronClust_spike_train':IronClustShare,
                     'HDSort_spike_train':HDSortShare}
    if Segname not in IndexName:
        Seg = ShareFunction.get(Segname)(dirname,RecordingParam,sampling_rate)
        blockdata.segments.append(Seg)

def GetEnsemble(ComparedSeg,SegnameList,blockdata):
    SegDict = {}
    
    for Seg in blockdata.segments:
        if Seg.name in SegnameList:
            SegDict[Seg.name]=Seg.spiketrains
                
    CompareDict = {}
    #for i in list(SegList.keys()):

    CompareDict[ComparedSeg] = []
    for j in SegDict.keys():
        if ComparedSeg==j:
            continue
        CompareDict[ComparedSeg].append(TwoSorterCompare(SegDict[ComparedSeg],SegDict[j]))
        
    if len(CompareDict[ComparedSeg])>4:
        SegNames = [i.name for i in blockdata.segments]
        MaxDict = {}
        for i in CompareDict.keys():
            MaxList = np.array([np.max(j,0) for j in CompareDict[i]])
            MaxList[MaxList>0.5] = 1
            MaxList[MaxList<0.5] = 0
            MaxList = MaxList.sum(0)
            MaxDict[i] = MaxList
        
        ComparedSegIndex = [n for n in range(len(SegNames)) if SegNames[n]==ComparedSeg][0]
        Seg = Segment(name = 'Ensemble_spike_train')
        for n,i in enumerate(blockdata.segments[ComparedSegIndex].spiketrains):
            if MaxDict[ComparedSeg][n]==0:
                continue
            if (MaxDict[ComparedSeg][n]<=2) and (i.description['snr']<5):
                continue
            spk_des = {'SimilarityLevel':MaxDict[ComparedSeg][n],
                       'SegID':n}
            i.description.update(spk_des)
            Seg.spiketrains.append(i)
            
        print('Cluster num:',len(Seg.spiketrains))
        blockdata.segments.append(Seg)
    else:
        print("not enough Seg")

class BaseShare():
    def __init__(self, raw_dirname='/home/cuilab/Desktop/Caeser_TDC/tdc_caeser202010153T001_lcy_peeler',
                 ComparedSeg='tdc_spike_train'):
    #%% create block file
        self.block_br = Block(name = 'NeuralData',
                          file_origin = raw_dirname,
                          file_datetime=time.asctime(time.localtime(time.time())))
   
        #self._set_channel_index()
        Seg = Segment(name='RecordingSystemEvent')
        event_time = blk_event_marker.get_event_timestamps()[0]/sampling_rate.magnitude*pq.sec
        event_marker = blk_event_marker.get_event_timestamps()[2]
        Seg.events.append(Event(event_time,labels=event_marker))    
        self.block_br.segments.append(Seg)
        
        self.mldata = BehaviorMetaDataInterface(raw_dirname).block
        
        dirnameList = [os.path.join(raw_dirname,'info.json'),
                       os.path.join(raw_dirname,'Phy'),
                       os.path.join(raw_dirname,'SpykingCircus_output.db'),
                       os.path.join(raw_dirname,'Herdingspikes_output.db'),
                       os.path.join(raw_dirname,'firings.mda'),
                       os.path.join(raw_dirname,'hdsort_output_results.mat')]
        
        SegnameList = ['tdc_spike_train',
                       'Kilo_spike_train',
                       'Spykingcircus_spike_train',
                       'Herdingspike2_spike_train',
                       'IronClust_spike_train',
                       'HDSort_spike_train']
        
        for i,j in zip(dirnameList,SegnameList):
            if not os.path.exists(i):
                continue
            print(i)
            print(j)
            BlockUpdate(i,j,self.block_br,RecordingParam,sampling_rate)
        
        #return
        #%% Ensemble
        GetEnsemble(ComparedSeg,SegnameList,self.block_br)
        SegNames = [i.name for i in self.block_br.segments]
        EnsembleIndex = [n for n in range(len(SegNames)) if SegNames[n]=='Ensemble_spike_train']
        #%%waveclus
        if len(EnsembleIndex)!=0:
            waveclus_dir = os.path.join(raw_dirname,'waveclus_output')
            dataio_dir = raw_dirname
            SelectedGroup = np.unique([i.description['group'] for i in self.block_br.segments[EnsembleIndex[0]].spiketrains])
            if os.path.exists(waveclus_dir): 
                shutil.rmtree(waveclus_dir)
                
            if not os.path.exists(waveclus_dir): 
                os.mkdir(waveclus_dir)
                Parallel(n_jobs=-1)(delayed(SaveMat)(i,dataio_dir,waveclus_dir) for i in tqdm(SelectedGroup))
            
    
            EnsembleData = self.block_br.segments[EnsembleIndex[0]].spiketrains
            EnsembleDataSpikeTrainID = [(ii.description['ElectrodeID'],ii.description['ElectrodeLabel']) for ii in EnsembleData]
            LFPSeg = Segment(name = 'LFP')
        
            ChnSelect = [Chn[0]-1 for Chn in EnsembleDataSpikeTrainID]
            ChnDescription = [RecordingParam[i] for i in ChnSelect]
            mdaList = LFPData['data'][ChnSelect].T     
            LFPsig = AnalogSignal(sampling_rate = LFPData['samp_per_s']*pq.Hz, 
                                  signal = mdaList*pq.uV,
                                  description = ChnDescription,
                                  name = 'LFPArray',
                                  t_start=0*pq.s)
            
            LFPSeg.analogsignals.append(LFPsig)
            
        else:
            SpikeTrainIndex = [n for n in range(len(SegNames)) if 'spike_train' in SegNames[n]]
            SpikeTrainData = self.block_br.segments[SpikeTrainIndex[0]].spiketrains
            SpikeTrainDataSpikeTrainID = [(ii.description['ElectrodeID'],ii.description['ElectrodeLabel']) for ii in SpikeTrainData.spiketrains]
            LFPSeg = Segment(name = 'LFP')
        
            ChnSelect = [Chn[0]-1 for Chn in SpikeTrainDataSpikeTrainID]
            ChnDescription = [RecordingParam[i] for i in ChnSelect]
            mdaList = LFPData['data'][ChnSelect].T     
            LFPsig = AnalogSignal(sampling_rate = LFPData['samp_per_s']*pq.Hz, 
                                  signal = mdaList*pq.uV,
                                  description = ChnDescription,
                                  name = 'LFPArray',
                                  t_start=0*pq.s)
            
            LFPSeg.analogsignals.append(LFPsig)
            
        self.block_br.segments.append(LFPSeg)
            
        #self._set_unit()        
        #%% get EMG & motion trajectory file
        EMG_path_list = glob.glob(raw_dirname+'/*csv')
        Motion_path_list = glob.glob(raw_dirname+'/*mat')
        
        if len(EMG_path_list)>0:
            self._EMG_segment(EMG_path_list[0])
        else:
            Seg = Segment(name = 'EMG')
            self.block_br.segments.append(Seg)
            
        if len(Motion_path_list)>0:
            bhv_name = glob.glob(raw_dirname+'/*bhv*')
            if len(bhv_name)!=0:
                bhv_name = glob.glob(raw_dirname+'/*bhv*')[0].split('/')[-1].split('.')
                Motion_path_list = [i for i in Motion_path_list if i.split('/')[-1].split('.')[0]!=bhv_name[0] and 'hdsort_output_results.mat' not in i]
            else:
                Motion_path_list = [i for i in Motion_path_list if 'hdsort_output_results.mat' not in i]
                
        if len(Motion_path_list)>0:
            self._Motion_segment(Motion_path_list[0])
        else:
            Seg = Segment(name = 'Motion')
            self.block_br.segments.append(Seg)
        self.save_block()
        
    def save_block(self):
        blockdata = {"block_br":self.block_br,"mldata":self.mldata}
        with open(os.path.join(self.raw_dirname,'block.db'),"wb") as f:
            joblib.dump(blockdata,f)
            
        #scio.savemat(os.path.join(raw_dirname,'block.mat'),blockdata)
            
    #%% set channel index
    '''
    def _set_channel_index(self):
        for i in self.dataio_br.channel_groups.keys():
            chx = ChannelIndex(name=self.dataio_br.channel_group_label(i),
                               index=i,
                               description=self.dataio_br.channel_groups[i]['geometry'],
                               channel_ids=self.dataio_br.channel_groups[i]['channels'],
                               channel_names=self.dataio_br.channel_group_path[i])
            self.block_br.channel_indexes.append(chx)'''       
            
    def _EMG_segment(self,EMG_path):
        Seg = Segment(name = 'EMG')
        EMG_data =  pd.read_csv(EMG_path,header=0,delimiter=',')
        EMG_keys =[i for i in list(EMG_data.keys()) if i.find('EMG')!=-1]
        ACC_keys =[i for i in list(EMG_data.keys()) if i.find('Acc')!=-1]
        for key in EMG_keys:
            EMG_sig = AnalogSignal(sampling_rate = 2000*pq.Hz, 
                                   signal = np.array(EMG_data[key])*pq.V,
                                   name = key,
                                   t_start=self.block_br.description['event_start'])
                
            Seg.analogsignals.append(EMG_sig)
            
        for key in ACC_keys:
            ACC_sig = AnalogSignal(sampling_rate = 1/0.00675*pq.Hz, 
                                   signal = np.array(EMG_data[key])*pq.gee,
                                   name = key,
                                   t_start=self.block_br.description['event_start'])
                
            Seg.analogsignals.append(ACC_sig)
        
        self.block_br.segments.append(Seg)
        
    def _Motion_segment(self,Motion_path):
        Seg = Segment(name = 'Motion')
        
        motion_data = hdf5storage.loadmat(Motion_path)['Unlabel']
        DataMaxLen = np.max([len(i[0]) for i in motion_data])
        motion_data = [i[0] for i in motion_data if len(i[0])==DataMaxLen]
        motion_time = [i for i in motion_data if i.shape[1]==1][0].squeeze()
        motion_time = motion_time-motion_time[0]
        motion_point = [i for i in motion_data if i.shape[1]==3][0]
        
        for ind, i in enumerate(motion_point.T):
            Mirsig = IrregularlySampledSignal(times = motion_time/(100*pq.Hz)+self.block_br.description['event_start'], 
                                              signal = i*pq.mm,
                                              description = '100hz',
                                              name = 'MotionCapture'+str(i),
                                              t_start = self.block_br.description['event_start'])
            
            Seg.irregularlysampledsignals.append(Mirsig)
        
        self.block_br.segments.append(Seg)
    

class BRshare():
    def __init__(self, raw_dirname='/home/cuilab/Desktop/Caeser_TDC/tdc_caeser202010153T001_lcy_peeler',
                 ComparedSeg='tdc_spike_train'):
        #%% Get NEV & ML data path  
        db_file = glob.glob(raw_dirname+'/block.db')
        if len(db_file)>0:
            with open(db_file[0],"rb") as f:
                blockdata = joblib.load(f)
            
            self.block_br = blockdata['block_br']
            self.mldata = blockdata['mldata']
            return
        self.raw_dirname = raw_dirname
        nev_dir = glob.glob(os.path.join(raw_dirname,'nev')+'/*.nev')[0]
        ns3File = glob.glob(raw_dirname+'/*.ns3')[0]
        ns3_file = NsxFile(ns3File)
        RecordingParam = ns3_file.extended_headers
        LFPData = ns3_file.getdata()
        ns3_file.close()
        nsFile = glob.glob(raw_dirname+'/*.ns6')[0]
        nsx_file = NsxFile(nsFile)
        RecordingParam = nsx_file.extended_headers
        sampling_rate = nsx_file.basic_header['TimeStampResolution']*pq.Hz
        nsx_file.close()
        
        blk_event_marker = io.BlackrockIO(nev_dir)
        
        
            

            
    
    
    
    
    
    
    
    
    
    
    
    
    
    