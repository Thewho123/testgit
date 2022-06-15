#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 22:03:01 2021

@author: cuilab
"""
import numpy as np
import quantities as pq
import glob
import scipy.io as scio
from neo import Block,Segment,Event,IrregularlySampledSignal,AnalogSignal,ImageSequence
from psychopy.tools.filetools import fromFile
import scipy
import time
import joblib

str2pq = {'1.0 s':pq.s,
          '1 s (second)':pq.s,
          '1.0 ms':pq.ms,
          '1.0 m':pq.m,
          '1.0 uV':pq.uV,
          '1 ms (millisecond)':pq.ms,
          '1.0 Hz':pq.Hz,
          '1.0 V':pq.V,
          '1.0 1/Hz':1/pq.Hz,
          '1.0 mm':pq.mm,
          'None':pq.dimensionless,
          '1.0 dimensionless':pq.dimensionless}

def GenerateMlDict(mldata):
    mldata_dict = {'name':'BehaviorData','Method':'MonkeyLogic','mldata':{}}
    mldata_dict['mldata']['Trial'] = [i[0][0].astype(float)-1 for i in mldata['Trial']]
    mldata_dict['mldata']['Block'] = [i[0][0].astype(float) for i in mldata['Block']]
    mldata_dict['mldata']['Condition'] = [i[0][0].astype(float) for i in mldata['Condition']]
    mldata_dict['mldata']['AbsoluteTrialStartTime'] = [i[0][0].astype(float) for i in mldata['AbsoluteTrialStartTime']]
    mldata_dict['mldata']['TrialError'] = [i[0][0].astype(float) for i in mldata['TrialError']]
    mldata_dict['mldata']['BehavioralCodes'] = {}
    
    mldata_dict['mldata']['BehavioralCodes']['CodeNumbers'] = []
    mldata_dict['mldata']['BehavioralCodes']['CodeTimes'] = []
    for i in mldata['BehavioralCodes']:
        CodeNumber = list(i['CodeNumbers'][0][0].squeeze().astype(float))
        CodeTime = list(i['CodeTimes'][0][0].squeeze().astype(float))
        DelIndex = [j for j in np.unique(CodeNumber) if CodeNumber.count(j) > 1]
        mldata_dict['mldata']['BehavioralCodes']['CodeNumbers'].append([j for j in CodeNumber if j not in DelIndex].copy())
        mldata_dict['mldata']['BehavioralCodes']['CodeTimes'].append([CodeTime[n] for n,j in enumerate(CodeNumber) if j not in DelIndex])    
    
    mldata_dict['mldata']['ObjectStatusRecord'] = {}
    mldata_dict['mldata']['ObjectStatusRecord']['Time'] = [list(i['Time'][0][0].squeeze().astype(float)) for i in mldata['ObjectStatusRecord']]
    mldata_dict['mldata']['ObjectStatusRecord']['Status'] = []
    for i in mldata['ObjectStatusRecord']:
        mldata_dict['mldata']['ObjectStatusRecord']['Status'].append([list(j.astype(float)) for j in list(i['Status'][0][0])])
    mldata_dict['mldata']['ObjectStatusRecord']['Position'] = []
    for i in mldata['ObjectStatusRecord']:
        EpochList = []
        for j in list(i['Position'][0][0]):
            EpochPos = j[0]
            EpochPos[np.isnan(EpochPos)] = 0
            EpochList.append([list(k.astype(float)) for k in EpochPos])
        mldata_dict['mldata']['ObjectStatusRecord']['Position'].append(EpochList)
    
    mldata_dict['mldata']['MarkTimeUnit'] = str(pq.ms)
    
    mldata_dict['mldata']['UserVars'] = {}
    UserVarNumMax = max([len(i.dtype.names) for i in mldata.squeeze()['UserVars']])
    for i in mldata.squeeze()['UserVars']:
        if len(i.dtype.names)==UserVarNumMax:
            UserVarName = i.dtype.names
            break
        
    for i in UserVarName:
        mldata_dict['mldata']['UserVars'][i] = []

    mldata_dict['mldata']['AnalogData'] = {}
    for AnalogKey in mldata.squeeze()[0]['AnalogData'].dtype.names:
        mldata_dict['mldata']['AnalogData'][AnalogKey] = []
    for Trial in mldata.squeeze(): 
        for anakey in Trial['AnalogData'].dtype.names:
            mldata_dict['mldata']['AnalogData'][anakey].append(Trial['AnalogData'][anakey][0][0])
        for UserKey in mldata_dict['mldata']['UserVars'].keys():
            if UserKey in Trial['UserVars'].dtype.names:
                mldata_dict['mldata']['UserVars'][UserKey].append(Trial['UserVars'][UserKey][0][0])
            else:
                mldata_dict['mldata']['UserVars'][UserKey].append([])
            #AnalogdataGridFS.Append(Trial['AnalogData'][anakey][0][0],{'name':anakey,'Trial':float(Trial['Trial'][0][0])})
    #mldata_dict['mldata']['AnalogData'] = AnalogdataGridFS.GridDict
    return mldata_dict

#psydata = fromFile('/mnt/PoissonData/tdc_Poisson20210605BrainControl1D001_Frank_peeler/Possion_2021-06-05-10-42-29-1DPSIDSharedMemory.psydat')
def GeneratePsyPyDict(psydata):
    mldata_dict = {'name':'BehaviorData','Method':'Psychopy3','mldata':{}}
    ColNum = (psydata.thisN+1)//psydata.data['order'].shape[0]
    mldata_dict['mldata']['Trial'] = list(psydata.data['order'][:,0:ColNum].flatten().astype(float))
    if 'Block' in psydata.data.keys():
        mldata_dict['mldata']['Block'] = list(psydata.data['Block'].data[:,0:ColNum].flatten().astype(float))
    if 'Condition' in psydata.data.keys():
        mldata_dict['mldata']['Condition'] = list(psydata.data['Condition'].data[:,0:ColNum].flatten().astype(float))
    if 'TrialStartTime' in psydata.data.keys():
        mldata_dict['mldata']['AbsoluteTrialStartTime'] = list(psydata.data['TrialStartTime'].data[:,0:ColNum].flatten().astype(float))
    if 'TrialError' in psydata.data.keys():
        mldata_dict['mldata']['TrialError'] = list(psydata.data['TrialError'].data[:,0:ColNum].flatten().astype(float))
    mldata_dict['mldata']['BehavioralCodes'] = {}
    
    mldata_dict['mldata']['BehavioralCodes']['CodeNumbers'] = []
    mldata_dict['mldata']['BehavioralCodes']['CodeTimes'] = []
    
    for i in psydata.data['EventMarker'][:,0:ColNum].flatten():
        CodeNumber = list(np.array(i)[:,0].astype(float))
        CodeTime = list(np.array(i)[:,1].astype(float)) 
        mldata_dict['mldata']['BehavioralCodes']['CodeNumbers'].append(CodeNumber)
        mldata_dict['mldata']['BehavioralCodes']['CodeTimes'].append(CodeTime)
        
    mldata_dict['mldata']['MarkTimeUnit'] = str(pq.s)
    #mldata_dict['mldata']['UserVars']['GridFS'] = {'GridFSKeyList':[],'GridFSValueList':[]}
    mldata_dict['mldata']['irr'] = {}
    mldata_dict['mldata']['UserVars'] = {}
    mldata_dict['mldata']['MarkTimeUnit'] = str(pq.s)
    for i in psydata.data['UserVars'][0,0].keys():
        if 'DecoderParameters' in i:
            mldata_dict['mldata']['UserVars']['DecoderParameters'] = []
            for j in psydata.data['UserVars'][:,0:ColNum].flatten():
                mldata_dict['mldata']['UserVars']['DecoderParameters'].append(j['DecoderParameters'])
            continue
        
        if isinstance(psydata.data['UserVars'][0,0][i],list):          
            if len(psydata.data['UserVars'][0,0][i])!=0:
                if isinstance(psydata.data['UserVars'][0,0][i][0],bool):
                    mldata_dict['mldata']['irr'][i] = [list(np.array(j[i]).astype(float))\
                                                       for j in psydata.data['UserVars'][:,0:ColNum].flatten()]
                else:
                    mldata_dict['mldata']['irr'][i] = [j[i] for j in psydata.data['UserVars'][:,0:ColNum].flatten()]
            else:
               mldata_dict['mldata']['irr'][i] = [j[i] for j in psydata.data['UserVars'][:,0:ColNum].flatten()]
        else:
            mldata_dict['mldata']['UserVars'][i] = [j[i] for j in psydata.data['UserVars'][:,0:ColNum].flatten()]
            
    mldata_dict['mldata']['irr']['AnalogData'] = list(psydata.data['AnalogData'][:,0:ColNum].flatten())
    return mldata_dict

def CollectDict(BehaviorData,ListIndex):
    description = {}
    for TrialDescriptionKey in BehaviorData['mldata']:
        if not isinstance(BehaviorData['mldata'][TrialDescriptionKey],list):
            continue
        
        description[TrialDescriptionKey] = BehaviorData['mldata'][TrialDescriptionKey][ListIndex]
        if 'Time' in TrialDescriptionKey and not isinstance(BehaviorData['mldata'][TrialDescriptionKey][ListIndex], str):
            description[TrialDescriptionKey] = description[TrialDescriptionKey]*str2pq[BehaviorData['mldata']['MarkTimeUnit']]
    return description

def AnaSeg(Seg,SubBehaviorData,SampleRate,AnaKey):
    Sig = AnalogSignal(sampling_rate = SampleRate*pq.Hz, 
                       signal = SubBehaviorData,
                       name = AnaKey,
                       t_start=0*pq.s)          
    Seg.analogsignals.append(Sig)

def IrrSeg(Seg,SubBehaviorData,SampleTime,irrKey):
    Units = SubBehaviorData.units
    if len(SubBehaviorData)!=len(SampleTime):
        if len(SubBehaviorData)==0:
            return
        SubBehaviorData = scipy.signal.resample(SubBehaviorData,len(SampleTime))*Units
    Sig = IrregularlySampledSignal(times = SampleTime,
                                   signal = SubBehaviorData,
                                   name = irrKey,
                                   t_start = 0*pq.s)            
    Seg.irregularlysampledsignals.append(Sig)

def UserVarSeg(Seg,ListIndex,SubBehaviorData):
    for UserVarKey in SubBehaviorData:
        try:
            if len(SubBehaviorData[UserVarKey][ListIndex])==0:
                continue          
        except TypeError:
            Seg.description.update({UserVarKey:SubBehaviorData[UserVarKey][ListIndex]})
            continue
        
        if isinstance(SubBehaviorData[UserVarKey][ListIndex], np.ndarray):
            if isinstance(SubBehaviorData[UserVarKey][ListIndex],str):
                Seg.description.update({UserVarKey:SubBehaviorData[UserVarKey][ListIndex]})
                continue
            if len(SubBehaviorData[UserVarKey][ListIndex].shape)==1:
                SubBehaviorData[UserVarKey][ListIndex] = SubBehaviorData[UserVarKey][ListIndex][:,np.newaxis]
            elif len(SubBehaviorData[UserVarKey][ListIndex].shape)==0:
                SubBehaviorData[UserVarKey][ListIndex] = np.array([[SubBehaviorData[UserVarKey][ListIndex]]])
            ImS = ImageSequence([SubBehaviorData[UserVarKey][ListIndex]],
                                name=UserVarKey,
                                spatial_scale=1 * pq.dimensionless,
                                units=pq.dimensionless,
                                index=0,
                                sampling_rate=1 * pq.Hz)
            Seg.imagesequences.append(ImS)
            continue
       
        for ind,i in enumerate(SubBehaviorData[UserVarKey][ListIndex]):
            if isinstance(i,str):
                Seg.description.update({UserVarKey:i})
                continue
            if len(i.shape)==1:
                i = i[:,np.newaxis]
            elif len(i.shape)==0:
                i = np.array([[i]])
                
            ImS = ImageSequence([i],
                                name=UserVarKey,
                                spatial_scale=1 * pq.dimensionless,
                                units=pq.dimensionless,
                                index=ind,
                                sampling_rate=1 * pq.Hz)
            Seg.imagesequences.append(ImS)


def LilabBehaviorMetaDataInterface(dirname):
    BehaviorData = scio.loadmat(dirname)
    key = [i for i in BehaviorData.keys() if '__' not in i][0]
    BehaviorData = BehaviorData[key]
    
    
class BehaviorMetaDataInterface():
    def __init__(self, raw_dirname,collection,mlpath=None): 
        if mlpath==None:
            mldataFile = glob.glob(raw_dirname+'/*.mat')
            try:
                bhv_name = glob.glob(raw_dirname+'/*bhv*')[0].split('/')[-1].split('.')[0]
                mldataFile = [i for i in mldataFile if i.split('/')[-1].split('.')[0]==bhv_name][0]
            except:
                bhv_name = ''
                mldataFile = []
        else:
            mldataFile = mlpath
                
        psy_name = glob.glob(raw_dirname+'/*psydat')
        if len(psy_name)!=0:
            block_psy = Block(name = collection,
                              file_origin = raw_dirname,
                              file_datetime=time.asctime(time.localtime(time.time())),
                              index=1)
            
            psydata = fromFile(psy_name[0])
            psydataDict = GeneratePsyPyDict(psydata)
            BlockDescription = {'Method':psydataDict['Method']}
            block_psy.description=BlockDescription
            
            for ind,i in enumerate(psydataDict['mldata']['Trial']):
                description = CollectDict(psydataDict,ind)
                Seg = Segment(index = int(i),
                              description = description,
                              name='BehaviorTrial')
                
                if hasattr(psydataDict['mldata']['BehavioralCodes']['CodeTimes'][ind], 'units'):
                    event_time = psydataDict['mldata']['BehavioralCodes']['CodeTimes'][ind]
                else:
                    event_time = psydataDict['mldata']['BehavioralCodes']['CodeTimes'][ind]*str2pq[psydataDict['mldata']['MarkTimeUnit']]
                
                event_marker = psydataDict['mldata']['BehavioralCodes']['CodeNumbers'][ind]
                Seg.events.append(Event(event_time,labels=event_marker))
                #duration = (event_time[-1]-event_time[0]).rescale(pq.s)
                #AnaSeg(Seg,ind,psydataDict['mldata']['AnalogData'],duration)
                if 'irr' in psydataDict['mldata'].keys():
                    if hasattr(psydataDict['mldata']['irr']['AnalogTime'][ind], 'units'):
                        SampleTime = psydataDict['mldata']['irr']['AnalogTime'][ind]
                    else:
                        SampleTime = psydataDict['mldata']['irr']['AnalogTime'][ind]*str2pq[psydataDict['mldata']['MarkTimeUnit']]
                        
                    for irrKey in psydataDict['mldata']['irr'].keys():
                        if hasattr(psydataDict['mldata']['irr'][irrKey][ind], 'units'):
                            IrrSeg(Seg,psydataDict['mldata']['irr'][irrKey][ind],SampleTime,irrKey)
                        else:
                            IrrSeg(Seg,psydataDict['mldata']['irr'][irrKey][ind]*pq.dimensionless,SampleTime,irrKey)
                            
                UserVarSeg(Seg,ind,psydataDict['mldata']['UserVars'])
                block_psy.segments.append(Seg)
            self.block = block_psy
                
        if len(mldataFile)!=0:
            block_ml = Block(name = collection,
                             file_origin = raw_dirname,
                             file_datetime=time.asctime(time.localtime(time.time())),
                             index=1)
            try:
                mldata = scio.loadmat(mldataFile)
                DataKey = [key for key in mldata if'__' not in key][0]
                mldata = mldata[DataKey][0]
            except:
                with open(mldataFile,'rb') as f:
                    mldata = joblib.load(f)
            mldataDict = GenerateMlDict(mldata) if not isinstance(mldata, dict) else mldata
            BlockDescription = {'Method':mldataDict['Method']}
            
            ObjStaRec = {}
            ObjStaRec['AnalogTime']=mldataDict['mldata']['ObjectStatusRecord']['Time']
            
            for ObjKey in mldataDict['mldata']['ObjectStatusRecord']:
                if 'Time' in ObjKey:
                    continue
                ObjStaRec[ObjKey] = mldataDict['mldata']['ObjectStatusRecord'][ObjKey]
                
            for ind,i in enumerate(mldataDict['mldata']['Trial']):
                description = CollectDict(mldataDict,ind)
                Seg = Segment(index = int(i),
                              description = description,
                              name='BehaviorTrial')
                
                if 'quantities' not in str(type(mldataDict['mldata']['BehavioralCodes']['CodeTimes'][ind])):
                    event_time = mldataDict['mldata']['BehavioralCodes']['CodeTimes'][ind]*str2pq[mldataDict['mldata']['MarkTimeUnit']]
                else:
                    event_time = mldataDict['mldata']['BehavioralCodes']['CodeTimes'][ind]
                    
                event_marker = mldataDict['mldata']['BehavioralCodes']['CodeNumbers'][ind]
                Seg.events.append(Event(event_time,labels=event_marker))
                
                duration = (event_time[-1]-event_time[0]).rescale(pq.s)
                
                if 'AnalogData' in mldataDict['mldata']:
                    for AnaKey in mldataDict['mldata']['AnalogData']:
                        if mldataDict['mldata']['AnalogData'][AnaKey][ind].dtype.names==None:
                            SampleRate = (len(list(mldataDict['mldata']['AnalogData'][AnaKey][ind]))*pq.dimensionless/duration).magnitude
                            AnaSeg(Seg,mldataDict['mldata']['AnalogData'][AnaKey][ind]*pq.dimensionless,SampleRate,AnaKey)
                                    
                        else:
                            for DtypeKey in mldataDict['mldata']['AnalogData'][AnaKey][ind].dtype.names:
                                SubSignal = mldataDict['mldata']['AnalogData'][AnaKey][ind][DtypeKey]
                                SampleRate = (len(list(SubSignal))*pq.dimensionless/duration).magnitude
                                AnaSeg(Seg,SubSignal*pq.dimensionless,SampleRate,AnaKey)
                            
                UserVarSeg(Seg,ind,mldataDict['mldata']['UserVars'])
                try:
                    SampleTime = ObjStaRec['AnalogTime'][ind]*str2pq[mldataDict['mldata']['MarkTimeUnit']]
                    for irrKey in ObjStaRec.keys():
                        IrrSeg(Seg,ObjStaRec[irrKey][ind]*pq.dimensionless,SampleTime,irrKey)
                except:
                    pass
                
                block_ml.segments.append(Seg)
                
            self.block = block_ml
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
