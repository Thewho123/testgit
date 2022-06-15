#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 22:24:46 2021

@author: cuilab
"""

import pymongo
import numpy as np
from gridfs import GridFS
from .BaseInterfaceTools import VariableDescriptionInterface
from neo import Block,Segment,SpikeTrain,Event,IrregularlySampledSignal,AnalogSignal,ImageSequence,Epoch
import quantities as pq

SegModuleDict = {'analogsignals':AnalogSignal,
                 'events':Event,
                 'imagesequences':ImageSequence,
                 'irregularlysampledsignals':IrregularlySampledSignal,
                 'spiketrains':SpikeTrain,
                 'epochs':Epoch}

SegDataName = ['analogsignals',
               'epochs',
               'events',
               'imagesequences',
               'irregularlysampledsignals',
               'spiketrains',
               'segments']

SubSegDataName = ['times',
                  'labels',
                  'sampling_rate',
                  't_start',
                  't_stop',
                  'file_origin',
                  'file_datetime',
                  'description',
                  'name',
                  'spatial_scale']

Generaltype = ['str',
               'ListDict',
               'int',
               'float',
               'dtype',
               'VariableArray']

def NeoIndex():
    n = 0
    while True:
        yield n
        n += 1             
            
class MongoNeoInterface():
    def __init__(self,collection,dbName="Poisson_monkey_array",
                MongoAdress="mongodb://localhost:27017/",
                username='admin', password='cuilab324'):
        #%% upload spiketrain
        myclient = pymongo.MongoClient(MongoAdress, username=username, password=password)
        self.mydb = myclient[dbName]
        self.collection = collection
        self.mycol = self.mydb[collection]
        self.fs = GridFS(self.mydb, collection=collection)
        self.vdi = VariableDescriptionInterface()
        if 'NeoIndex' not in self.mycol.index_information():
            self.mycol.create_index([('blocksIndex',pymongo.ASCENDING),
                                     ('segmentsIndex',pymongo.ASCENDING),
                                     ('spiketrainsIndex',pymongo.ASCENDING),
                                     ('imagesequencesIndex',pymongo.ASCENDING),
                                     ('epochsIndex',pymongo.ASCENDING),
                                     ('eventsIndex',pymongo.ASCENDING),
                                     ('irregularlysampledsignalsIndex',pymongo.ASCENDING),
                                     ('analogsignalsIndex',pymongo.ASCENDING)],unique=True,name='NeoIndex')
        self.indexDict = {}   
        self.indexDictGenerator = {}  
        
    def Sent2Mongo(self,blockdata):
        List = []
        self._Encoder(blockdata,List,blockdata.index)
        self.mycol.insert_many(List)
        
    def _Encoder(self,Neodata,List,BlockIndex):
        self.indexDict['blocksIndex'] = BlockIndex
        for neoattr in set(SegDataName) & set(dir(Neodata)):
            Data = getattr(Neodata,neoattr)
            self.indexDictGenerator[neoattr+'Index'] = NeoIndex()
            self.indexDict[neoattr+'Index'] = -1
            
            SubSegData = {neoattr+'Index':self.indexDict[neoattr+'Index'],
                          'blocksIndex':BlockIndex,
                          'segmentsIndex':self.indexDict['segmentsIndex']}
            for j in set(SubSegDataName) & set(dir(Data)):
                if getattr(Data,j)==None: continue
                if 'analogsignals' in neoattr and 'times' in j: continue
                SubSegData[j] = self.vdi.encode(getattr(Data,j),self.fs)
            
            for i in Data:
                self.indexDict[neoattr+'Index'] = next(self.indexDictGenerator[neoattr+'Index'])
                if hasattr(i,'index'):
                    if i.index!=None: self.indexDict[neoattr+'Index'] = i.index
                self._Encoder(i,List,BlockIndex)
                
        SubSegData = {Neodata.__module__.split('.')[-1]+'sIndex':self.indexDict[Neodata.__module__.split('.')[-1]+'sIndex'],
                          'blocksIndex':BlockIndex,
                          'segmentsIndex':self.indexDict['segmentsIndex']}
        
        if 'block' in Neodata.__module__.split('.')[-1]: del SubSegData['segmentsIndex']
        
        for j in set(SubSegDataName) & set(dir(Neodata)):            
            try:
                if getattr(Neodata,j)==None and not isinstance(getattr(Neodata,j), np.ndarray):
                    continue
            except ValueError:
                pass 
            self.debuger = getattr(Neodata,j)
            SubSegData[j] = self.vdi.encode(getattr(Neodata,j),self.fs)
            
        if len(set(SegDataName) & set(dir(Neodata)))==0:
            SubSegData['NeoData'] = self.vdi.encode(Neodata,self.fs)
            
        if len(set(SegDataName) & set(dir(Neodata)))!=0 or len(set(SubSegDataName) & set(dir(Neodata)))!=0:
            List.append(SubSegData)
    
    def _Decoder(self,SegName,BlockIndex, NotLoadIndexList = [], LFP=False):
        
        SegmentIndex = {}
        IndexList = [i+'Index' for i in SegDataName if 'segments' not in i]
        for SegAttr in SegDataName:
            SegmentIndex[SegAttr+'Index'] = {'$exists' :'segments' in SegAttr}
        
        for BlockInfo in self.mycol.find({'segmentsIndex':{'$exists' : False},
                                          'blocksIndex':BlockIndex},{'_id': 0}):
            
            BlockInfo['index'] = BlockInfo['blocksIndex']
            blockdata = Block(**self.vdi.decode(BlockInfo))
            SegmentIndex['blocksIndex'] = blockdata.index
            TempDict = {}
            for SegInfo in self.mycol.find(SegmentIndex,{'_id': 0}).sort([('segmentsIndex', 1)]):
                SegInfo['index'] = SegInfo['segmentsIndex']
                Seg = Segment(**self.vdi.decode(SegInfo))
                blockdata.segments.append(Seg)
                
                TempDict[SegInfo['index']] = {}
                for subkeys in SegDataName:
                    if "segments" in subkeys: continue
                    TempDict[SegInfo['index']][subkeys] = []
            
            SegAttrIndex = {}
            SegAttrIndex['blocksIndex'] = blockdata.index
            for NotLoadIndex in NotLoadIndexList:
                SegAttrIndex[NotLoadIndex+'Index'] = {'$exists' :False}
                
            for SegAttr in self.mycol.find(SegAttrIndex,{'_id': 0}):
                SegListName = SegAttr.keys() & set(IndexList)
                if len(SegListName)==0: continue                   
                    
                if 'LFP' in blockdata.segments[SegAttr['segmentsIndex']].name and not LFP: continue
                # if blockdata.segments[SegAttr['segmentsIndex']].name not in SegName and\
                    # 'spiketrainsIndex' in SegListName: continue
                SegListName = SegListName.pop() 
                SegAttrInfo = self.vdi.decode(SegAttr,self.fs)
                SegAttrInfo['index'] = SegAttrInfo[SegListName]
                
                if 'spike_train' in blockdata.segments[SegAttr['segmentsIndex']].name or 'eventsIndex' in SegListName: 
                    if 'times' in SegAttrInfo: 
                        del SegAttrInfo['times']
                
                if 'str' in SegAttrInfo['NeoData'].dtype.name:
                    blockdata.segments[SegAttr['segmentsIndex']].description[SegAttrInfo['name']] = SegAttrInfo['NeoData'].squeeze()
                    continue
                
                if 'irregularlysampledsignalsIndex' in SegListName: 
                    SegAttrInfo['signal'] = SegAttrInfo['NeoData']
                    if 'times' not in SegAttrInfo: SegAttrInfo['times'] = SegAttrInfo['signal']
                    del SegAttrInfo['NeoData']
                    SubNeoModuleData = SegModuleDict.get(SegListName[0:-5])(**SegAttrInfo)
                else:
                    SubNeoModuleData = SegModuleDict.get(SegListName[0:-5])(SegAttrInfo['NeoData'],
                                                              **SegAttrInfo)
                SubNeoModuleData.index=SegAttrInfo['index']
                TempDict[SegAttr['segmentsIndex']][SegListName[0:-5]].append(SubNeoModuleData)
                    
            for SegIndex in TempDict:
                for subarr in TempDict[SegIndex]:
                    if len(TempDict[SegIndex][subarr])==0: continue
                    SubSegList = getattr(blockdata.segments[SegIndex],subarr)
                    TempDict[SegIndex][subarr].sort(key=lambda x: x.index)
                    for ele in TempDict[SegIndex][subarr]: SubSegList.append(ele)

        return blockdata         
                
            
                
                
                
                
                
                
                
                