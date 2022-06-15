#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:12:33 2021

@author: cuilab
"""

import pymongo
import os
from share_multi import tdcshare
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["Caesar_monkey"]
import numpy as np
from tqdm import tqdm
import quantities as pq
from tridesclous import DataIO
import glob
import json


if __name__ == '__main__':
    raw_dir = '/home/cuilab/Desktop/CaeserData'
    #peeler_dirlist = [i for i in os.listdir(raw_dir) if (i.find('peeler')!=-1) and (i.find('3T')==-1) and (len(i.split('_'))==5)]
    peeler_dirlist = [i for i in os.listdir(raw_dir) if (i.find('peeler')!=-1)]
    geomPath = '/home/cuilab/Desktop/Caeser_TDC/geom.csv'
    with open(geomPath,'r') as f: geom = np.loadtxt(f,delimiter=',')
    GeomList = [list(i.astype(float)) for i in geom]
    
    for dir_name in tqdm(peeler_dirlist):   
        collist = mydb.list_collection_names()
        dataio_dir = os.path.join(raw_dir,dir_name)
        raw_file = glob.glob(os.path.join(dataio_dir+'/mda','*.raw'))
        info_path = glob.glob(os.path.join(dataio_dir,'*.json'))[0]
        with open(info_path,'r',encoding='utf-8')as fp:
            json_data = json.load(fp)
        
        json_data['datasource_kargs']['filenames'] = raw_file
        
        os.remove(info_path)
        with open(info_path,'w',encoding='utf-8') as json_file:
            json.dump(json_data,json_file)
        dataio_br = DataIO(dirname=dataio_dir)
        '''
        if dir_name in collist:
            continue
            print("collection "+dir_name+" existed")'''
        blockdata = tdcshare(os.path.join(raw_dir,dir_name))
        mycol = mydb[dir_name]
        mldata = blockdata.mldata
        mldata_dict = GenerateMlDict(mldata)
        InsertResult = mycol.insert_one(mldata_dict)
        geom_dict = {'name':'geom','geom':GeomList}
        InsertResult = mycol.insert_one(geom_dict)
        #FinalIndex = [n for n,i in enumerate(blockdata.block_br.segments) if i.name=='FinalEnsemble_spike_train'][0]
        for Seg in blockdata.block_br.segments:
            FinalBlockSpikeTrains = Seg.spiketrains
            if len(FinalBlockSpikeTrains)==0:
                continue
            for st in FinalBlockSpikeTrains:
                SpikeTrainDict = {'units':str(st.units),
                                  'SegName':Seg.name,
                                  'SampleRate':dataio_br.sample_rate,
                                  'SampleRateUnit':str(pq.Hz)}
                SampleRate = dataio_br.sample_rate*pq.Hz
                DescriptionDict = {}
                if st.description!=None:
                    DescriptionDict = st.description.copy()
                    
                    if 'par' in DescriptionDict.keys():
                        del DescriptionDict['par']
                    
                    if 'mean_waveform' in DescriptionDict.keys():
                        DescriptionDict['mean_waveform'] = list(DescriptionDict['mean_waveform'].astype(float))
                    
                    for key in DescriptionDict.keys():
                        try:
                            DescriptionDict[key] = DescriptionDict[key].astype(float)
                        except:
                            try:
                                DescriptionDict[key] = float(DescriptionDict[key])
                            except:
                                pass
                    
                SpikeTrains = st.times.rescale(pq.s)
                SpikeTrains = SpikeTrains*SampleRate
                SpikeTrainDict.update({'Description':DescriptionDict})
                SpikeTrainDict.update({'SpikeTimes':(SpikeTrains.magnitude).astype(np.int32).tobytes()})
                InsertResult = mycol.insert_one(SpikeTrainDict)
                '''
                try:
                    InsertResult = mycol.insert_one(SpikeTrainDict)
                except:
                    #SpikeTimes = st.times.rescale(pq.ms)
                    SpikeTrainDict['SpikeTimes'] = (st.times.magnitude).tobytes()
                    SpikeTrainDict['units'] = str(pq.ms)
                    InsertResult = mycol.insert_one(SpikeTrainDict)'''
                    
                
            
        
