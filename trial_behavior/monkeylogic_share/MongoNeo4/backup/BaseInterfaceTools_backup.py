#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 22:03:01 2021

@author: cuilab
"""

import quantities as pq
import numpy as np
import sys
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

str2dtype = {str(float):float,
             'float64':np.float64,
             str(np.float):np.float,
             str(np.float32):np.float32,
             str(np.float64):np.float64,
             'uint8' : np.int8}
#%% Array interface
def Array2Bytes(Array):
    InsertDict = {}
    InsertDict['Value'] = Array.tobytes()
    InsertDict['Dtype'] = str(Array.dtype)
    InsertDict['Shape'] = Array.shape
    return InsertDict

def Units2Dict(UnitsArray):
    InsertDict = {}
    InsertDict['magnitude'] = UnitsArray.magnitude.astype(float)
    InsertDict['units'] = str(UnitsArray.units)
    return InsertDict
    
    
def Bytes2Array(InsertDict):
    return np.frombuffer(InsertDict['Value'],dtype=np.dtype(InsertDict['Dtype'])).reshape(InsertDict['Shape'])

class VariableArrayInterface():
    def encode(self,Array,fs=None):
        EncodedArray = {}
        if hasattr(Array, 'units'):
            Array = Units2Dict(Array)
            Array['magnitude']=Array2Bytes(Array['magnitude'])
        else:
            Array=Array2Bytes(Array)
        
        if fs!=None:
            gif = GridInterface()
            if 'magnitude' in Array.keys():
                if len(Array['magnitude']['Value'])>=1600000:
                    gif.Append(Array['magnitude']['Value'], {})
                    gif.Put(fs)
                    Array['magnitude']['Value'] = gif.GridDict
            else:
                if len(Array['Value'])>=1600000:
                    gif.Append(Array['Value'], {})
                    gif.Put(fs)
                    Array['Value'] = gif.GridDict
        EncodedArray['VariableArray'] = Array
        
        return EncodedArray
    
    def decode(self,Dict,fs=None):
        assert isinstance(Dict, dict), 'Please send a dict'
        assert 'VariableArray' in Dict.keys(), 'Please send a array dict'
        GFSBrunch = False
        if 'magnitude' in Dict['VariableArray'].keys():
            if not isinstance(Dict['VariableArray']['magnitude']['Value'],bytes):
                GFSBrunch=True
        else:
            if not isinstance(Dict['VariableArray']['Value'],bytes):
                GFSBrunch=True
        if fs!=None and GFSBrunch:
            if 'magnitude' in Dict['VariableArray'].keys():
                if 'GridFS' in Dict['VariableArray']['magnitude']['Value']:
                    gif = GridInterface(Dict['VariableArray']['magnitude']['Value'])
                    gif.Get(fs)
                    Dict['VariableArray']['magnitude']['Value'] = \
                                  Dict['VariableArray']['magnitude']['Value']['GridFS']['GridFSValueList'][0]
                else:
                     raise BaseException("Can not find 'GridFS' Key") 
            else:           
                if 'GridFS' in Dict['VariableArray']['Value']:
                    gif = GridInterface(Dict['VariableArray']['Value'])
                    gif.Get(fs)
                    Dict['VariableArray']['Value'] = Dict['VariableArray']['Value']['GridFS']['GridFSValueList'][0]
                else:
                    raise BaseException("Can not find 'GridFS' Key") 
                                  
        if 'magnitude' in Dict['VariableArray'].keys():
            return Bytes2Array(Dict['VariableArray']['magnitude'])*str2pq[Dict['VariableArray']['units']]
        else:
            return Bytes2Array(Dict['VariableArray'])
#%% description interface
def Array4Mongo(Array,Dict,Data2Mongo,fs=None):
    vai = VariableArrayInterface()
    if Array.dtype.names!=None:
        for dtypeKey in Array.dtype.names:
            Dict[dtypeKey] = {}
            Data2Mongo(Array[dtypeKey],Dict[dtypeKey])
    elif Array.dtype.name=='object':
        Data2Mongo(list(Array),Dict)
    
    else:
        Dict.update(vai.encode(Array,fs))
        
def Data2Mongo(description,DescriptionDict,fs=None):
    
    if isinstance(description,list):
        if len(description)==1:
            Data2Mongo(description[0],DescriptionDict)
        else:
            DescriptionDict['ListDict'] = [{} for _ in range(len(description))]
            for ChnDescription,ListDict in zip(description,DescriptionDict['ListDict']):
                Data2Mongo(ChnDescription,ListDict)
    
    elif isinstance(description,dict):
        for key in description.keys():
            DescriptionDict[key] = {}
            if len(description)!=0:
                Data2Mongo(description[key],DescriptionDict[key])
    
    elif isinstance(description,np.ndarray): Array4Mongo(description,DescriptionDict,Data2Mongo,fs)
    elif isinstance(description,int): DescriptionDict.update({'int':float(description)})
    elif isinstance(description,str): DescriptionDict.update({'str':description})
    elif isinstance(description,float): DescriptionDict.update({'float':description})
    elif hasattr(description, 'dtype'): DescriptionDict.update({'dtype':{description.dtype.name:float(description)}})
    else:
        if hasattr(description, '__module__'):
            DescriptionDict[description.__class__.__name__] = {}
            Data2Mongo(description.__dict__,DescriptionDict[description.__class__.__name__])
        else:
            raise Exception("please give a list or dict containing array, float, int or str")
    
def Mongo2PY(DescriptionDict,description,fs=None):
    vai = VariableArrayInterface()
    if 'ListDict' in DescriptionDict.keys(): 
        description['ListDict'] = []
        for i in range(len(DescriptionDict['ListDict'])):
            description['ListDict'] = [{}]*len(DescriptionDict['ListDict'])
            Mongo2PY(DescriptionDict['ListDict'][i],description['ListDict'][i])
        
    elif 'int' in DescriptionDict.keys(): description['int'] = int(DescriptionDict['int'])
    elif 'VariableArray' in DescriptionDict.keys(): description['VariableArray'] = vai.decode(DescriptionDict,fs)
    elif 'float' in DescriptionDict.keys(): description['float'] = DescriptionDict['float']
    elif 'str' in DescriptionDict.keys(): description['str'] = DescriptionDict['str']
    elif 'dtype' in DescriptionDict.keys(): 
        DataTypeName = list(DescriptionDict['dtype'].keys())[0]
        description['dtype'] = np.array([DescriptionDict['dtype'][DataTypeName]]).astype(np.dtype(DataTypeName))[0]
    
    else:
        for Key in DescriptionDict.keys():
            if isinstance(DescriptionDict[Key],dict):
                if 'ListDict' in DescriptionDict[Key].keys(): 
                    description[Key] = [{} for _ in range(len(DescriptionDict[Key]['ListDict']))]
                    for i in range(len(DescriptionDict[Key]['ListDict'])):
                        Mongo2PY(DescriptionDict[Key]['ListDict'][i],description[Key][i])
                    
                elif 'int' in DescriptionDict[Key].keys(): description[Key] = int(DescriptionDict[Key]['int'])
                elif 'VariableArray' in DescriptionDict[Key].keys(): description[Key] = vai.decode(DescriptionDict[Key],fs)
                elif 'float' in DescriptionDict[Key].keys(): description[Key] = DescriptionDict[Key]['float']
                elif 'str' in DescriptionDict[Key].keys(): description[Key] = DescriptionDict[Key]['str']
                else:
                    description[Key] = {}
                    Mongo2PY(DescriptionDict[Key],description[Key])
            else:
                description[Key] = DescriptionDict[Key]
    
class VariableDescriptionInterface():
    def encode(self,description,fs=None):
        DescriptionDict = {}
        Data2Mongo(description,DescriptionDict,fs)
        return DescriptionDict
    def decode(self,DescriptionDict,fs=None):
        description = {}
        Mongo2PY(DescriptionDict,description,fs)
        return description
#%% GridInterface
class GridInterface:
    def __init__(self,GridDict=None):
        if GridDict == None:
            self._GridDict = {'GridFS':{'GridFSKeyList':[],'GridFSValueList':[]}}
        else:
            self._GridDict = GridDict
            
    def Append(self,AnalogDataValue,AnalogDataKey):
        if isinstance(AnalogDataValue,list):
            AnalogDataKey['GridFS']['type'] = 'str'
            self._GridDict['GridFS']['GridFSValueList'].append(str(AnalogDataValue))
            self._GridDict['GridFS']['GridFSKeyList'].append(AnalogDataKey)
        elif isinstance(AnalogDataValue,bytes):
            AnalogDataKey['type'] = 'bytes'
            self._GridDict['GridFS']['GridFSValueList'].append(AnalogDataValue)
            self._GridDict['GridFS']['GridFSKeyList'].append(AnalogDataKey)
        else:
            raise TypeError('Wrong AnalogDataValue type, the input must be list or numpy.ndarray, the type is',type(AnalogDataValue))
    
    def Put(self,fs):
        #GridInput = self._GridDict['GridFS']['GridFSValueList']
        for ElementKey, GridInputElement in zip(self._GridDict['GridFS']['GridFSKeyList'],self._GridDict['GridFS']['GridFSValueList']):
            #self._GridDict['GridFS']['GridFSKeyList'][Index]['_id'] = fs.put(GridInputElement,**ElementKey)
            if '_id' in ElementKey:
                continue
            if isinstance(GridInputElement, bytes):
                ElementKey['_id'] = fs.put(GridInputElement,**ElementKey)           
            else:
                ElementKey['_id'] = fs.put(GridInputElement.encode(),**ElementKey)
                
        del self._GridDict['GridFS']['GridFSValueList']
        
    def Get(self,fs):
        self._GridDict['GridFS']['GridFSValueList'] = []
        for Key in self._GridDict['GridFS']['GridFSKeyList']:
            out = fs.get(Key['_id']).read()
            if Key['type'] == 'bytes':
                self._GridDict['GridFS']['GridFSValueList'].append(out)
            elif Key['type'] == 'str':
                outList = out.decode().split('],')
                self._GridDict['GridFS']['GridFSValueList'].append(list(map(lambda x: [float(i) for i in x.split(',')                                                                                if '[' not in i and ']' not in i], outList)))
    @property
    def GridDict(self):
        return self._GridDict
def GenerateQuery():
    pass