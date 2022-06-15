#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 22:03:01 2021

@author: cuilab
"""

import quantities as pq
import numpy as np
import sys 
from affinewarp import SpikeData
from affinewarp.spikedata import is_sorted
import datetime

Generaltype = ['str',
               'ListDict',
               'int',
               'float',
               'dtype',
               'VariableArray']

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

def SearchAndUpload(func):
    
    def WarpedFunction(self,InsertDict):
        StatisticsResults = self._ReadOut(InsertDict)
        if isinstance(StatisticsResults,dict): return StatisticsResults['UploadData']
        Data = func(self)
        self._Upload(InsertDict,Data)
        return self._ReadOut(InsertDict)['UploadData']
        
    return WarpedFunction

def StatisticsResults2SpikeData4DP(StatisticsResults,spiketrains,kargsList):
    data = SpikeData4DP(
        trials=StatisticsResults['SpikeData'][0],
        spiketimes = StatisticsResults['SpikeData'][1]*spiketrains[0].units.rescale(pq.ms).magnitude.squeeze(),
        neurons=StatisticsResults['SpikeData'][2],
        tmin=0,
        tmax=(kargsList[0]['t_stop']-kargsList[0]['t_start']).rescale(pq.ms).magnitude.squeeze() if kargsList[0]['aligned_marker']!=kargsList[0]['aligned_marker2']\
            else max([i['t_stop']-i['t_start'] for i in kargsList]).rescale(pq.ms).magnitude.squeeze()
    )

    for i in StatisticsResults:
        if 'SpikeData' in i: continue
        setattr(data,i,StatisticsResults[i])
        
    return data

class SpikeData4DP(SpikeData):
    def __init__(
            self, trials, spiketimes, neurons, tmin, tmax,
            n_trials=None, n_neurons=None):
        '''
        A structure inheriting 'SpikeData' from 'affinewarp'.
        Because of different sequence rule, a new structure 'SpikeData4DP' is built.
        The spiketrain time sorting is achieved by 'spike_time' method of 'DP'.
        So, I remove the np.lexsort part and the sequence is equal to np.lexsort((spiketimes, neurons, trials))
        Just to be sure, in the following development, NOT USING THE SEQUENCE BUT USING SEARCHING!!
        The spiketimes is aligned to 'tmin', so you can change 'tmin' of 'tmax' to get the\
            slicing of spiketimes in the BORDER OF A TRIAL!!

        Parameters
        ----------
        trials : np.array
            sequence containing trials ind.
        spiketimes : Tnp.array
            sequence containing spiketimes ind.
        neurons : np.array
            sequence containing neurons ind.
        tmin : float
            The left part of the spiketime.
        tmax : float
            The left part of the spiketime.
        n_trials : TYPE, optional
            If can be caculated by itself. The default is None.
        n_neurons : TYPE, optional
            If can be caculated by itself. The default is None.

        Returns
        -------
        None.

        '''
        
        # Treat inputs as numpy arrays.
        self.trials = np.ascontiguousarray(trials, dtype=int)
        self.spiketimes = np.ascontiguousarray(spiketimes, dtype=float).ravel()
        self.neurons = np.ascontiguousarray(neurons, dtype=int).ravel()
        self.tmin = tmin
        self.tmax = tmax
        self.n_neurons = int(np.max(neurons) + 1 if n_neurons is None else n_neurons)
        self.n_trials = int(np.max(trials) + 1 if n_trials is None else n_trials)
        self._frac_spiketimes = None
    def select_trials(self, kept_trials, inplace=False):
        """
        Filter out trials by integer id.
        """
        if not np.iterable(kept_trials):
            kept_trials = (kept_trials,)
        kept_trials = np.asarray(kept_trials)
        if kept_trials.dtype == bool:
            kept_trials = np.where(kept_trials)[0]
        elif not is_sorted(kept_trials):
            raise ValueError("kept_trials must be sorted.")
        result = self if inplace else self.copy()
        result._filter(result.trials.copy(), kept_trials)
        result.sort_spikes()
        result.n_trials = result.trials[-1] + 1
        return result

#%% Array interface
def Array2Bytes(Array):
    InsertDict = {}
    InsertDict['Value'] = Array.tobytes()
    InsertDict['Dtype'] = str(Array.dtype)
    InsertDict['Shape'] = Array.shape
    return InsertDict

def Units2Dict(UnitsArray):
    InsertDict = {}
    #InsertDict['magnitude'] = UnitsArray.magnitude.astype(float)
    InsertDict['magnitude'] = UnitsArray.magnitude
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
                if sys.getsizeof(Array['magnitude']['Value'])>=(15000000/2):
                    gif.Append(Array['magnitude']['Value'], {})
                    gif.Put(fs)
                    Array['magnitude']['Value'] = gif.GridDict
            else:
                if sys.getsizeof(Array['Value'])>=(15000000/2):
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
            try:
                return Bytes2Array(Dict['VariableArray']['magnitude'])*str2pq[Dict['VariableArray']['units']]
            except:
                return Bytes2Array(Dict['VariableArray']['magnitude'])
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
            Data2Mongo(description[0],DescriptionDict,fs)
        else:
            DescriptionDict['ListDict'] = [{} for _ in range(len(description))]
            for ChnDescription,ListDict in zip(description,DescriptionDict['ListDict']):
                Data2Mongo(ChnDescription,ListDict,fs)
    
    elif isinstance(description,dict):
        for key in description.keys():
            DescriptionDict[key] = {}
            if len(description)!=0:
                Data2Mongo(description[key],DescriptionDict[key],fs)
    
    elif isinstance(description,np.ndarray): Array4Mongo(description,DescriptionDict,Data2Mongo,fs)
    elif isinstance(description,int): DescriptionDict.update({'int':float(description)})
    elif isinstance(description,str): DescriptionDict.update({'str':description})
    elif isinstance(description,float): DescriptionDict.update({'float':description})
    elif isinstance(description,datetime.date): DescriptionDict.update({'str':str(description)})
    elif hasattr(description, 'dtype'): DescriptionDict.update({'dtype':{description.dtype.name:float(description)}})
    else:
        if hasattr(description, '__module__'):
            DescriptionDict[description.__class__.__name__] = {}
            Data2Mongo(description.__dict__,DescriptionDict[description.__class__.__name__])
        elif description==None:
            DescriptionDict.update({'str':str(None)})
        else:
            raise Exception("please give a list or dict containing array, float, int or str")
    
def Mongo2PY(vai,DescriptionDict,description,fs=None):
    #vai = VariableArrayInterface()
    if isinstance(description,list):
        if 'ListDict' in DescriptionDict.keys(): 
            description.append([])
            for i in range(len(DescriptionDict['ListDict'])):
                Mongo2PY(vai,DescriptionDict['ListDict'][i],description[-1],fs)
            
        elif 'int' in DescriptionDict.keys(): description.append(int(DescriptionDict['int']))
        elif 'VariableArray' in DescriptionDict.keys(): description.append(vai.decode(DescriptionDict,fs))
        elif 'float' in DescriptionDict.keys(): description.append(DescriptionDict['float'])
        elif 'str' in DescriptionDict.keys(): description.append(DescriptionDict['str'])
        elif 'dtype' in DescriptionDict.keys(): 
            DataTypeName = list(DescriptionDict['dtype'].keys())[0]
            description.append(np.array([DescriptionDict['dtype'][DataTypeName]]).astype(np.dtype(DataTypeName))[0])
        else:
            description.append({})
            Mongo2PY(vai,DescriptionDict,description[-1],fs)        
    else:
        if 'ListDict' in DescriptionDict.keys(): 
            description['ListDict'] = []
            for i in range(len(DescriptionDict['ListDict'])):
                Mongo2PY(vai,DescriptionDict['ListDict'][i],description['ListDict'],fs)
            
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
                        description[Key] = []
                        for i in range(len(DescriptionDict[Key]['ListDict'])):
                            Mongo2PY(vai,DescriptionDict[Key]['ListDict'][i],description[Key],fs)
                        
                    elif 'int' in DescriptionDict[Key].keys(): description[Key] = int(DescriptionDict[Key]['int'])
                    elif 'VariableArray' in DescriptionDict[Key].keys(): description[Key] = vai.decode(DescriptionDict[Key],fs)
                    elif 'float' in DescriptionDict[Key].keys(): description[Key] = DescriptionDict[Key]['float']
                    elif 'str' in DescriptionDict[Key].keys(): description[Key] = DescriptionDict[Key]['str']
                    elif 'dtype' in DescriptionDict[Key].keys(): 
                        DataTypeName = list(DescriptionDict[Key]['dtype'].keys())[0]
                        description[Key] = np.array([DescriptionDict[Key]['dtype'][DataTypeName]]).astype(np.dtype(DataTypeName))[0]
                    else:
                        description[Key] = {}
                        Mongo2PY(vai,DescriptionDict[Key],description[Key],fs)
                else:
                    description[Key] = DescriptionDict[Key]

class VariableDescriptionInterface():
    def encode(self,description,fs=None):
        DescriptionDict = {}
        Data2Mongo(description,DescriptionDict,fs)
        return DescriptionDict
    def decode(self,DescriptionDict,fs=None):
        description = {}
        self.vai = VariableArrayInterface()
        Mongo2PY(self.vai,DescriptionDict,description,fs)
        for key in description.keys():
            if not isinstance(description[key], dict): continue
            if list(description[key].keys())[0] in Generaltype:
                description[key] = description[key][list(description[key].keys())[0]]               
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