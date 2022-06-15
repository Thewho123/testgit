#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:04:14 2020

@author: sorter
"""

from numba import cuda
import numpy as np
import math
from tqdm import tqdm
from ..MongoReader import MongoReadModule
import quantities as pq
from elephant.kernels import GaussianKernel
import cupy
from tridesclous.iotools import ArrayCollection
import os
import shutil
import time
import json
import pymongo
from ..BaseInterfaceTools import GenerateQuery
from itertools import chain
#Tempdir = '/home/cuilab/Desktop/Caeser_TDC/CCHTemp'
def NWInterface(SpikeTime):
    TrialInd=np.unique(SpikeTime.trials)
    CellInd=np.unique(SpikeTime.neurons)
    SpikeInput=[]
    for tri in tqdm(TrialInd):
        CellList=[]
        for cell in CellInd:
            CellList.append(SpikeTime.spiketimes[(SpikeTime.trials==tri)*(SpikeTime.neurons==cell)]*SpikeTime.annotation['units'])
        SpikeInput.append(CellList)
    return SpikeInput
#%% some func for space comunication 
def CreateArrayJson(CollectionName,ArrayList,Tempdir,
                    rep = 1000):

    JSONDICT = {}
    TempdirCollection = os.path.join(Tempdir,CollectionName)
    for i in range(int(rep)):
        JSONDICT.update({str(i): {
                                    "dtype": "float32",
                                    "shape": ArrayList,
                                    "annotations": {}}})
    JSONPATH = os.path.join(TempdirCollection,'arrays.json')
    if os.path.exists(JSONPATH):
        os.remove(JSONPATH)
    with open(JSONPATH,'w') as f:
        json.dump(JSONDICT,f)
        
def delete_file(filePath):
    
    if os.path.exists(filePath):
        if os.path.isfile(filePath):
            os.remove(filePath)
            try:
                shutil.rmtree((filePath))
            except:
                pass
            
            return
        
        try:
            shutil.rmtree((filePath))
        except:
            pass
        
        try:
            for filename in os.listdir(filePath):
                if os.path.isfile(filename):
                    os.remove(filename)
                try:
                    shutil.rmtree((filePath))
                except:
                    pass
                        
                else:
                    delete_file(os.path.join(filePath,filename))
        except:
            pass

def free_gpu_memory():
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
def mkcollectiondir(CollectionList,Tempdir):
    
    for CollectionName in CollectionList:
        TempdirCollection = os.path.join(Tempdir,CollectionName)
        if os.path.exists(TempdirCollection):
            delete_file(TempdirCollection)

#%% Generate serugate spike train
@cuda.jit
def CUDAstsGeneration(GenerateSpike,stsGenerated,d_SpikeLength):
    a,b = cuda.grid(2)
    if (a<stsGenerated.shape[0]) and (b<stsGenerated.shape[1]):
        for i in range(GenerateSpike[a,:,b].shape[0]):
            if GenerateSpike[a,i,b] > 0:
                stsGenerated[a,b,int(d_SpikeLength[a,b])] = i+1
                d_SpikeLength[a,b] = d_SpikeLength[a,b]+1
        
                
def MainCUDAStsGenerated(GenerateSpike,d_FiringRate,d_sts,d_SpikeLength):
    GenerateSpike[:] = cupy.random.poisson(d_FiringRate)
    d_sts[:] = np.nan
    d_SpikeLength[:] = 0
    
    TPB = 32
    threadsperblock = (TPB,TPB)
    blockspergrid_x = math.ceil(d_sts.shape[0]/threadsperblock[0])
    blockspergrid_y = math.ceil(d_sts.shape[0]/threadsperblock[1])
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    CUDAstsGeneration[blockspergrid, threadsperblock](GenerateSpike,d_sts,d_SpikeLength)
    cuda.synchronize()
#%% find cch 
@cuda.jit
def CUDATrialccg(sts,CCH,SpikeLength,
                 nbins,tbin):
    
    a = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    b = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    Trial = cuda.blockIdx.z*cuda.blockDim.z + cuda.threadIdx.z
    if (a<sts.shape[1]) and (Trial<sts.shape[0]) and (b<sts.shape[1]):
        st1 = sts[Trial,a,0:SpikeLength[Trial,a]]
        st2 = sts[Trial,b,0:SpikeLength[Trial,b]]
        ihigh = 0
        ilow = 0
        j = 0
        dt = nbins*tbin*1000

        while j<st2.size:
            while (ihigh<st1.size) and (st1[int(ihigh)] < st2[j]+dt):
                ihigh = ihigh + 1
                
            while (ilow<st1.size) and (st1[int(ilow)] <= st2[j]-dt):
                ilow = ilow + 1
                
            if ilow>=st1.size: break
            if st1[ilow] > st2[j]+dt:
                j = j+1
                continue
            
            k = ilow
            while k<ihigh:
                ibin = round(((st2[j] - st1[k])/tbin)/1000)
                cuda.atomic.add(CCH, (a,b,int(ibin + nbins)), 1.0)

                k = k+1
            j = j+1
        
def CUDAMainTrialCcg(d_sts,d_CCH,d_SpikeLength,nbins,tbin):
    #TPB = 6
    d_CCH[:] = 0
    threadsperblock = (10,10,10)
    blockspergrid_x = math.ceil(d_sts.shape[1]/threadsperblock[0])
    blockspergrid_y = math.ceil(d_sts.shape[1]/threadsperblock[1])
    blockspergrid_z = math.ceil(d_sts.shape[0]/threadsperblock[2])
    blockspergrid = (blockspergrid_x,blockspergrid_y,blockspergrid_z)
    CUDATrialccg[blockspergrid, threadsperblock](d_sts,d_CCH,d_SpikeLength,
                                                 nbins,tbin)
    cuda.synchronize() 

def CUDAMainRescaledTrialCcg(SpikeLength,stsGenerated,nbins,tbin,st1sec,st2sec):
    d_SpikeLength = cupy.asarray(SpikeLength,dtype = cupy.float32)
    d_sts = cupy.asarray(stsGenerated,dtype = cupy.float32)
    d_CCH = cupy.zeros((d_sts.shape[1],d_sts.shape[1],2*nbins+1),dtype = cupy.float32)
    
    MeanSpikeLength = SpikeLength.mean(0)
    MeanSpikeLengthArray = np.sqrt(MeanSpikeLength[:,np.newaxis]@MeanSpikeLength[np.newaxis,:]/(st1sec*st2sec))
    MeanSpikeLengthArray = np.repeat(MeanSpikeLengthArray[:,:,np.newaxis],d_CCH.shape[-1],axis=-1)
    
    CUDAMainTrialCcg(d_sts,d_CCH,d_SpikeLength,nbins,tbin)
    
    NTaoVector = np.concatenate((np.arange(0,nbins),np.arange(nbins,-1,-1)))
    NTaoVector[[0,-1]] = 1
    NTaoMat = np.repeat(NTaoVector[np.newaxis,:],d_CCH.shape[1],axis=0)
    NTaoMat = np.repeat(NTaoMat[np.newaxis,:,:],d_CCH.shape[0],axis=0)
    CCH = ((cupy.asnumpy(d_CCH)/NTaoVector)/MeanSpikeLengthArray)/d_sts.shape[0]
    return CCH

#%% get surrogate cch and save it to memory map
def CCHCorection(FiringRate,
                 CollectionName,
                 Tempdir,
                 nbins=500,
                 tbin=0.001,
                 repNum = 5,
                 repBlock = 100,
                 st1sec = 2,
                 st2sec = 2,
                 DeleteFile = True):
    
    FiringRate[FiringRate<0] = 0
    FiringRate = np.round(FiringRate/1000,6)
    
    d_FiringRate = cupy.asarray(FiringRate,dtype=cupy.float32)
    GenerateSpike = cupy.asarray(FiringRate,dtype=cupy.float32)
    CCH = np.zeros((FiringRate.shape[-1],FiringRate.shape[-1],2*nbins+1,repBlock))
    
    d_CCH = cupy.asarray(CCH,dtype=cupy.float32)
    
    stsGenerated = np.zeros((FiringRate.shape[0],FiringRate.shape[-1],FiringRate.shape[1]))
    d_sts = cupy.asarray(stsGenerated,dtype=cupy.float32)  # d_ --> device
    
    SpikeLength = np.zeros((repBlock,FiringRate.shape[0],FiringRate.shape[-1]))
    d_SpikeLength = cupy.asarray(SpikeLength,dtype=cupy.float32)    
    TempdirCollection = os.path.join(Tempdir,CollectionName)
    
    if DeleteFile and (os.path.exists(TempdirCollection)):
        delete_file(TempdirCollection)
    NTaoVector = np.concatenate((np.arange(0,nbins),np.arange(nbins,-1,-1)))
    NTaoVector[[0,-1]] = 1
            
    TimeStart = time.time()
    arrays = ArrayCollection(parent=None, dirname=TempdirCollection)
    for j in repNum:
        for i in tqdm(range(repBlock)):
            MainCUDAStsGenerated(GenerateSpike,d_FiringRate,d_sts,d_SpikeLength[i])        
            CUDAMainTrialCcg(d_sts,d_CCH[:,:,:,i],d_SpikeLength[i],nbins,tbin)
            
        CCHResults = cupy.asnumpy(d_CCH)
        
        for m in range(repBlock):
            MeanSpikeLength = cupy.asnumpy(d_SpikeLength[m]).mean(0)
            #MeanSpikeLength[MeanSpikeLength==0] = 1
            MeanSpikeLengthArray = np.sqrt(MeanSpikeLength[:,np.newaxis]@MeanSpikeLength[np.newaxis,:]/(st1sec*st2sec))
            MeanSpikeLengthArray = np.repeat(MeanSpikeLengthArray[:,:,np.newaxis],d_CCH.shape[2],axis=-1)
            CCHResults[:,:,:,m] = ((CCHResults[:,:,:,m]/NTaoVector)/MeanSpikeLengthArray)/d_sts.shape[0]
            
        for k in range(CCHResults.shape[-1]):
            arrays.create_array(str(j*repBlock+k),
                                np.float32,
                                (CCHResults.shape[0],CCHResults.shape[1],CCHResults.shape[2]),
                                'memmap')
            RawDataArray = arrays.get(str(j*repBlock+k))
            RawDataArray[:] = CCHResults[:,:,:,k]
            arrays.detach_array(str(j*repBlock+k))
    print(time.time()-TimeStart)
#%% get raw cch
def cudaFiringRateCCH(spike_time,
                      nbins=500,
                      tbin=0.001,
                      st1sec = 2,
                      st2sec = 2):
    
    TrialNum = len(spike_time)
    CellNum = len(spike_time[0])
    stsGenerated = np.zeros((TrialNum,CellNum,1000))
    stsGenerated[:] = np.nan
    SpikeLength = np.zeros((TrialNum,CellNum))
    
    for n,tri in enumerate(spike_time):
        for m,cell in enumerate(tri):
            SpikeNum = cell.shape[0]
            if SpikeNum==0:continue
            SpikeLength[n,m] = SpikeNum
            stsGenerated[n,m,0:SpikeNum] = np.array(cell.rescale(pq.s))*1000
            
    CCH = CUDAMainRescaledTrialCcg(SpikeLength,stsGenerated,nbins,tbin,st1sec,st2sec)
    return CCH
#%% cuda find cluster zescore value & position of peak
def CUDAMainZscore(d_CCH,d_CCH_Mean,d_CCH_Std):
    for i in range(d_CCH.shape[-1]):
        d_CCH[:,:,i] = (d_CCH[:,:,i]-d_CCH_Mean)/d_CCH_Std

@cuda.jit
def CUDAFindCluster(CorrectedZscoreCCH,PeakIndex,ClusterResults):
    a = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    b = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    
    if (a<CorrectedZscoreCCH.shape[0]) and (b<CorrectedZscoreCCH.shape[1]):
        for i in range(1,CorrectedZscoreCCH.shape[-1]):   
            if CorrectedZscoreCCH[a,b,i]>=2:
                if CorrectedZscoreCCH[a,b,i-1]<2:
                    PeakIndex[a,b] = PeakIndex[a,b]+1
                ClusterResults[a,b,i,0] = PeakIndex[a,b]
                ClusterResults[a,b,int(PeakIndex[a,b]),1] = ClusterResults[a,b,int(PeakIndex[a,b]),1]+CorrectedZscoreCCH[a,b,i]
        
        for i in range(1,CorrectedZscoreCCH.shape[-1]):   
            if CorrectedZscoreCCH[a,b,i]<=-2:
                if CorrectedZscoreCCH[a,b,i-1]>-2:
                    PeakIndex[a,b] = PeakIndex[a,b]+1
                ClusterResults[a,b,i,0] = PeakIndex[a,b]
                ClusterResults[a,b,int(PeakIndex[a,b]),1] = ClusterResults[a,b,int(PeakIndex[a,b]),1]+CorrectedZscoreCCH[a,b,i]

def CUDAMainFindCluster(d_CCH,d_PeakIndex,d_ClusterResults):
    TPB = 32
    threadsperblock = (TPB,TPB)
    blockspergrid_x = math.ceil(d_CCH.shape[0]/threadsperblock[0])
    blockspergrid_y = math.ceil(d_CCH.shape[1]/threadsperblock[1])
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    CUDAFindCluster[blockspergrid, threadsperblock](d_CCH,d_PeakIndex,d_ClusterResults)

#%% Permutation test
@cuda.jit
def CUDAGetSelectMeans(d_Observation,d_RandpermSet,d_SelectMeans):
    a = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    b = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    c = cuda.blockIdx.z*cuda.blockDim.z + cuda.threadIdx.z
    if (a<d_RandpermSet.shape[0]) and (b<d_RandpermSet.shape[1]) and (c<d_RandpermSet.shape[2]):
        d_SelectMeans[a,b,c] =  d_Observation[d_RandpermSet[a,b,c],b,c]

@cuda.jit
def CUDAFindpValue(d_ObservedDifference,d_RandDifference,d_p):
    a = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    b = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    if (a<d_ObservedDifference.shape[0]) and (b<d_ObservedDifference.shape[1]):
        for i in range(d_RandDifference.shape[0]):
            if abs(d_ObservedDifference[a,b]) < abs(d_RandDifference[i,a,b]):
                d_p[a,b] = d_p[a,b]+1

def CCHZClusterPermutationTest(CCHRawClusterResults,CCHSurrogateClusterResults,permutation = 1000):
    Observation = np.concatenate((CCHSurrogateClusterResults,CCHSurrogateClusterResults.swapaxes(1,2)),axis=0)
    Observation = np.concatenate((CCHRawClusterResults[np.newaxis,:,:],Observation),axis=0)
    SamplePermutation = np.repeat(Observation.sum(0)[np.newaxis,:,:],repeats=permutation,axis =0)
    
    d_Observation = cupy.asarray(Observation,dtype=cupy.float32)
    d_RandpermSet = cupy.random.randint(0,Observation.shape[0],SamplePermutation.shape)
    d_SelectMeans = cupy.zeros(SamplePermutation.shape,dtype = cupy.float32)
    
    threadsperblock = (10,10,10)
    blockspergrid_x = math.ceil(d_RandpermSet.shape[0]/threadsperblock[0])
    blockspergrid_y = math.ceil(d_RandpermSet.shape[1]/threadsperblock[1])
    blockspergrid_z = math.ceil(d_RandpermSet.shape[2]/threadsperblock[2])
    blockspergrid = (blockspergrid_x,blockspergrid_y,blockspergrid_z)
    CUDAGetSelectMeans[blockspergrid, threadsperblock](d_Observation,d_RandpermSet,d_SelectMeans)
    cuda.synchronize()
    
    SelectMeans = cupy.asnumpy(d_SelectMeans)
    SamplePermutationMean = (SamplePermutation-SelectMeans)/(Observation.shape[0]-1)
    RandDifference = SelectMeans-SamplePermutationMean
    ObservedDifference = CCHRawClusterResults-Observation[1::].mean(0)
    
    d_ObservedDifference = cupy.asarray(ObservedDifference,dtype = cupy.float32)
    d_RandDifference = cupy.asarray(RandDifference,dtype = cupy.float32)
    d_p = cupy.zeros(d_ObservedDifference.shape,dtype = cupy.float32)
    
    TPB = 32
    threadsperblock = (TPB,TPB)
    blockspergrid_x = math.ceil(d_ObservedDifference.shape[0]/threadsperblock[0])
    blockspergrid_y = math.ceil(d_ObservedDifference.shape[1]/threadsperblock[1])
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    CUDAFindpValue[blockspergrid, threadsperblock](d_ObservedDifference,d_RandDifference,d_p)
    cuda.synchronize()
    
    d_p = (d_p+1)/(permutation + 1)
    p = cupy.asnumpy(d_p)
    
    return p

#%% FDR
@cuda.jit
def CUDAFDR(d_ClusterNum,d_psort,d_ConnectMatrix):
    a = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    b = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    c = cuda.blockIdx.z*cuda.blockDim.z + cuda.threadIdx.z
    if (a<d_ClusterNum.shape[0]) and (b<d_ClusterNum.shape[1]) and (c <= d_ClusterNum[a,b]):
        if d_ClusterNum[a,b]>0:
            for i in range(-1*int(d_ClusterNum[a,b]),0):
                if d_psort[a,b,i] <= 0.01:
            #   if d_psort[a,b,i] <= (0.05 * (i+d_ClusterNum[a,b]+1) / d_ClusterNum[a,b]):
                    d_ConnectMatrix[a,b] = 1

def PFDR(ClusterNum,psort):
    d_ClusterNum = cupy.asarray(ClusterNum,dtype=cupy.float32)
    d_psort = cupy.asarray(psort,dtype=cupy.float32)
    d_ConnectMatrix = cupy.zeros(d_ClusterNum.shape,dtype=cupy.float32)
    
    threadsperblock = (10,10,10)
    blockspergrid_x = math.ceil(d_ClusterNum.shape[0]/threadsperblock[0])
    blockspergrid_y = math.ceil(d_ClusterNum.shape[1]/threadsperblock[1])
    blockspergrid_z = math.ceil(ClusterNum.max()/threadsperblock[2])
    blockspergrid = (blockspergrid_x,blockspergrid_y,blockspergrid_z)
    CUDAFDR[blockspergrid, threadsperblock](d_ClusterNum,d_psort,d_ConnectMatrix)
    cuda.synchronize()
    return cupy.asnumpy(d_ConnectMatrix)

@cuda.jit
def CUDARemoveNegAxis(ClusterResults,p):
    a = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    b = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    c = cuda.blockIdx.z*cuda.blockDim.z + cuda.threadIdx.z
    if (a<ClusterResults.shape[0]) and (b<ClusterResults.shape[1]) and (c<int(ClusterResults.shape[2]/2)):
        if ClusterResults[a,b,c,0]!=0:
            if ClusterResults[a,b,c,0]!=ClusterResults[a,b,c+1,0]:
                p[a,b,int(ClusterResults[a,b,c,0])] = 0

def CUDARemoveNegAxisMain(ClusterResults,p):
    threadsperblock = (10,10,10)
    blockspergrid_x = math.ceil(ClusterResults.shape[0]/threadsperblock[0])
    blockspergrid_y = math.ceil(ClusterResults.shape[1]/threadsperblock[1])
    blockspergrid_z = math.ceil(int(ClusterResults.shape[2]/2)/threadsperblock[2])
    blockspergrid = (blockspergrid_x,blockspergrid_y,blockspergrid_z)
    d_ClusterResults = cupy.asarray(ClusterResults,dtype=cupy.float32)
    d_p = cupy.asarray(p,dtype=cupy.float32)
    CUDARemoveNegAxis[blockspergrid, threadsperblock](d_ClusterResults,d_p)
    cuda.synchronize()
    return cupy.asnumpy(d_p)
#%% build cch network finder module
class CCHModule(MongoReadModule):
    def CCHConnectedMatrix(self,**kargs):
        kargs['SegName'] = self.SegName
        self.CCHkargs = kargs
        InsertDict = kargs.copy()
        '''
        GenerateQuery(InsertDict,kargs)
        InsertDict['name'] = 'CCHNetWork'
        
        if 'DeleteQuery' in kargs.keys():
            if kargs['DeleteQuery']:
                self.mycol.delete_many(InsertDict)
        
        if self.mycol.find_one(InsertDict)!=None: return [i for i in self.mycol.find(InsertDict)]'''
        
        self._SurrogateCCH(Statistics = 'instantaneous_rate',
                           sampling_period = kargs['sampling_period'], 
                           Tempdir = kargs['Tempdir'],
                           t_left = kargs['t_left']-kargs['CCHcutoff'], 
                           t_right = kargs['t_right']+kargs['CCHcutoff'],
                           kernel = kargs['kernel'],
                           aligned_marker = kargs['aligned_marker'],
                           TrialError=kargs['TrialError'],
                           nbins=kargs['nbins'],
                           tbin=kargs['tbin'],
                           repNum = kargs['repNum'],
                           repBlock = kargs['repBlock'],
                           st1sec = kargs['st1sec'],
                           st2sec = kargs['st2sec'],
                           CCHcutoff = kargs['CCHcutoff'],
                           Clustercutoff = kargs['Clustercutoff'],
                           StatisticsML = kargs['FRStatisticsML'])

        self._RawCCH(Statistics = 'spike_time' ,
                     t_left = kargs['t_left'], 
                     t_right = kargs['t_right'],
                     aligned_marker = kargs['aligned_marker'],
                     TrialError=kargs['TrialError'],
                     nbins=kargs['nbins'],
                     tbin=kargs['tbin'],
                     st1sec = kargs['st1sec'],
                     st2sec = kargs['st2sec'],
                     Clustercutoff = kargs['Clustercutoff'],
                     StatisticsML = kargs['STStatisticsML'])
        
        self._FindConnectedMatrix(kargs['Clustercutoff'],kargs['permutation'])
        if kargs['DeleteFile']:
            delete_file(os.path.join(self.Tempdir,self.collection))
        '''
        InsertDict['ConnectedMatrix'] = {}
        InsertDict['ConnectedMatrix']['bytes'] = self.ConnectMatrix.tobytes()
        InsertDict['ConnectedMatrix']['shape'] = self.ConnectMatrix.shape
        InsertDict['ConnectedMatrix']['dtype'] = str(self.ConnectMatrix.dtype)
        
        GroupIndexList = [sts.description['group']\
                          for n,sts in enumerate(self.block_br.segments[0].spiketrains)\
                              if self.CellSelected[n]]
        InsertDict['GroupIndexList'] = GroupIndexList
        GroupName = [sts.description['ElectrodeLabel']\
                     for n,sts in enumerate(self.block_br.segments[0].spiketrains)\
                         if self.CellSelected[n]]
            
        InsertDict['GroupName'] = GroupName
        self.mycol.insert_one(InsertDict.copy())
        return [i for i in self.mycol.find(InsertDict)]'''
        
    def _SurrogateCCH(self,**kargs):
        self.Tempdir = kargs['Tempdir']
        TempdirCollection = os.path.join(self.Tempdir,self.collection)
        assert 'instantaneous_rate' in kargs['Statistics'],'please input instantaneous_rate'
        
        THparList = ['Statistics','sampling_period','t_left',
                     't_right','kernel','aligned_marker','TrialError','DeleteQuery']
        IRkargs = {}
        for i in kargs:
            if i in THparList:
                IRkargs[i] = kargs[i]
                
        self.FiringRate = np.array(self.SpikeStatisticsReader(**IRkargs)).swapaxes(1, -1)
        
        CutoffIndex = int(kargs['CCHcutoff'].rescale(kargs['sampling_period'].units).magnitude)

        self.FiringRate = self.FiringRate[:,CutoffIndex:(self.FiringRate.shape[1]-CutoffIndex+1),:]
        self.CellSelected = self.FiringRate.sum(1).mean(0)>100
        self.FiringRate = self.FiringRate[:,:,self.CellSelected]
        print('FiringRate Shape',self.FiringRate.shape)
        
        self.SurrogateKargs = kargs
        
        if os.path.exists(TempdirCollection):
            #self._CreateNFArrayJson()
            self._GetMeanSurrogateCCH()
            return
            #delete_file(os.path.join(self.Tempdir,self.collection))
        else:
            CCHCorection(FiringRate = self.FiringRate,
                         CollectionName = self.collection,
                         Tempdir = self.Tempdir,
                         nbins=kargs['nbins'],
                         tbin=kargs['tbin'],
                         repNum = kargs['repNum'],
                         repBlock = kargs['repBlock'],
                         st1sec = kargs['st1sec'],
                         st2sec = kargs['st2sec'])
        
        free_gpu_memory()
        rep = kargs['repNum'].shape[0]*kargs['repBlock']
        self._CreateNFArrayJson(rep)
        self._GetMeanSurrogateCCH()
        
    def _RawCCH(self,**kargs):
        assert 'spike_time' in kargs['Statistics'],'please input spike_time'
        STparList = ['Statistics','t_left',
                     't_right','aligned_marker','TrialError','DeleteQuery']
        STkargs = {}
        for i in kargs:
            if i in STparList:
                STkargs[i] = kargs[i]
        self.debuger=STkargs
        SpikeTime = self.SpikeStatisticsReader(**STkargs)
        self.spike_time = NWInterface(SpikeTime)
        STIndex = np.unique(SpikeTime.neurons)
        print('spike_time length',len(self.spike_time))
        
        TempSpikeTime = []
        for i in self.spike_time:
            TempSpikeTime.append([j for n,j in enumerate(i) if self.CellSelected[n] or n not in STIndex])
            
        self.spike_time = TempSpikeTime
        self.CollectionRawCCH = cudaFiringRateCCH(self.spike_time,
                                                  nbins=kargs['nbins'],
                                                  tbin=kargs['tbin'],
                                                  st1sec = kargs['st1sec'],
                                                  st2sec = kargs['st2sec'])
        self.RawCCHKargs = kargs
        free_gpu_memory()
        self.RawClusterResults = self._RawClusterBasedTest(self.CollectionRawCCH,kargs['Clustercutoff'])
        free_gpu_memory()
        
    def _GetMeanSurrogateCCH(self):
        
        arrays = ArrayCollection(parent=None, dirname=os.path.join(self.Tempdir,self.collection))
        arrays.load_all()
        SurrogateCCH = arrays.get(list(arrays.keys())[0])
        self.MeanSurrogateCCH = np.zeros(SurrogateCCH.shape)
        pbar = tqdm(list(arrays.keys()))
        for i in pbar:
            self.MeanSurrogateCCH[:] = self.MeanSurrogateCCH[:]+arrays.get(i)/len(arrays.keys())
            pbar.set_description("Processing GetMeanSurrogateCCH:")
        for i in pbar:
            arrays.detach_array(i)
        
        
    def _SurrogateClusterBasedTest(self,CCH,Clustercutoff):
        
        CorrectedCCH = CCH - self.MeanSurrogateCCH
        d_CCH = cupy.asarray(CorrectedCCH[:,:,Clustercutoff-1:-Clustercutoff],dtype=cupy.float32)
        d_CCH_Mean = d_CCH[:,:,1::].mean(-1)
        d_CCH_Std = d_CCH[:,:,1::].std(-1,ddof=1)
        CUDAMainZscore(d_CCH,d_CCH_Mean,d_CCH_Std)
        cuda.synchronize()
        
        d_CCH[:,:,0] = 0
        d_PeakIndex = cupy.zeros((d_CCH.shape[0],d_CCH.shape[1]),dtype=cupy.float32)
        d_ClusterResults = cupy.zeros(d_CCH.shape+(2,),dtype=cupy.float32)
        CUDAMainFindCluster(d_CCH,d_PeakIndex,d_ClusterResults)
        cuda.synchronize()
        
        d_MaxZscroeArray = cupy.abs(d_ClusterResults[:,:,1::,1]).max(-1)
        ClusterResults = cupy.asnumpy(d_MaxZscroeArray)
        free_gpu_memory()
        return ClusterResults
    
    def _RawClusterBasedTest(self,CCH,Clustercutoff):
        
        CorrectedCCH = CCH - self.MeanSurrogateCCH
        d_CCH = cupy.asarray(CorrectedCCH[:,:,Clustercutoff-1:-Clustercutoff],dtype=cupy.float32)
        d_CCH_Mean = d_CCH[:,:,1::].mean(-1)
        d_CCH_Std = d_CCH[:,:,1::].std(-1,ddof=1)
        CUDAMainZscore(d_CCH,d_CCH_Mean,d_CCH_Std)
        
        d_CCH[:,:,0] = 0
        d_PeakIndex = cupy.zeros((d_CCH.shape[0],d_CCH.shape[1]),dtype=cupy.float32)
        d_ClusterResults = cupy.zeros(d_CCH.shape+(2,),dtype=cupy.float32)
        CUDAMainFindCluster(d_CCH,d_PeakIndex,d_ClusterResults)
        cuda.synchronize()
        
        return cupy.asnumpy(d_ClusterResults)
        
    def _SurrogateCluster(self,Clustercutoff):
        TempdirCollection = os.path.join(self.Tempdir,self.collection)
        arrays = ArrayCollection(parent=None, dirname=TempdirCollection)
        arrays.load_all()
        SurrogateClusterList = []
        pbar = tqdm(list(arrays.keys()))
        for i in pbar:
            SurrogateCCH = arrays.get(i)
            SurrogateClusterList.append(self._SurrogateClusterBasedTest(SurrogateCCH,Clustercutoff))
            free_gpu_memory()
            pbar.set_description("Processing FindSurrogateCluster:")
            arrays.detach_array(i)
        self.SurrogateClusterArray = np.array(SurrogateClusterList)
        
    def _PermutationTest(self,permutation = 1000):
        CCHRawClusterResults = self.RawClusterResults[:,:,:,1]
        #CCHRawClusterIndex = self.RawClusterResults[:,:,:,0]
        CCHSurrogateClusterResults = self.SurrogateClusterArray
        self.PermutationTestpValue = np.zeros(CCHRawClusterResults.shape)
        print('Start PermutationTest')
        for i in range(CCHRawClusterResults.shape[-1]):
            if np.abs(CCHRawClusterResults[:,:,i]).sum() == 0:
                continue
            
            self.PermutationTestpValue[:,:,i] = CCHZClusterPermutationTest(np.abs(CCHRawClusterResults[:,:,i]),
                                                                           CCHSurrogateClusterResults,
                                                                           permutation)
            free_gpu_memory()
        
        ZeroRawObervation = np.abs(CCHRawClusterResults)==0
        self.PermutationTestpValue[ZeroRawObervation] = 0
        
        
    def _FDR(self):
        p = CUDARemoveNegAxisMain(self.RawClusterResults,self.PermutationTestpValue)
        ClusterNum = (p>0).sum(-1)
        psort = np.sort(p,-1)

        self.ConnectMatrix = PFDR(ClusterNum,psort)
        self.ConnectMatrix[range(self.ConnectMatrix.shape[0]),
                           range(self.ConnectMatrix.shape[0])] = 0
    
    def _FindConnectedMatrix(self,Clustercutoff,permutation = 1000):
        self._SurrogateCluster(Clustercutoff)
        self._PermutationTest(permutation)
        self._FDR()
    
    def _CreateNFArrayJson(self,rep=1000):
        CreateArrayJson(self.collection,
                        [self.FiringRate.shape[-1],
                         self.FiringRate.shape[-1],
                         2*self.SurrogateKargs['nbins']+1],
                        self.Tempdir,
                        rep=rep)
#%% main func
if __name__ == '__main__':
    myclient = pymongo.MongoClient("mongodb://10.10.47.78:27017/", username='admin', password='cuilab324')
    mydb = myclient["Caesar_monkey_array"]
    #mkcollectiondir(CollectionList,Tempdir)
    CollectionList = mydb.list_collection_names()
    CollectionList = [i for i in CollectionList if i.find('11')==-1 and i.find('3T')==-1]
    #ListChunk = int(len(CollectionList)/3)
    #CollectionList = CollectionList[0:ListChunk]
    
    dtypeList = {}
    dtypeList['float32'] = np.float32
    dtypeList['float64'] = np.float64
    
    
    NetworkDict = {}
    for CollectionName in CollectionList:
    
        NetWorkFinder = CCHModule(collection = CollectionName,
                                  SegName = 'FinalEnsemble_spike_train',
                                  Saverip="mongodb://10.10.47.78:27017/",
                                  db='Caesar_monkey_array',
                                  username='yongxiang',
                                  password='cuilab322',
                                  LFP=False)

        if '3T' in CollectionName:
            continue
        
        aligned_marker = 5
        t_left = -300*pq.ms
        t_right = 500*pq.ms
        kernel = GaussianKernel(3.66*pq.ms)
        CCHcutoff = 200*pq.ms
        
        ConnectedMatrix = NetWorkFinder.CCHConnectedMatrix(sampling_period = 1*pq.ms, 
                                                           Tempdir = '/home/cuilab/CCHTemp',
                                                           t_left = t_left, 
                                                           t_right = t_right,
                                                           kernel = kernel,
                                                           aligned_marker = aligned_marker,
                                                           TrialError=0,
                                                           nbins=500,
                                                           tbin=0.001,
                                                           repNum = np.arange(0,10),
                                                           repBlock = 100,
                                                           st1sec = 0.8,
                                                           st2sec = 0.8,
                                                           CCHcutoff = 200*pq.ms,
                                                           Clustercutoff = 300,
                                                           FRStatisticsML = None,
                                                           STStatisticsML = None,
                                                           permutation = 1000,
                                                           DeleteFile = True)
        NetworkDict[CollectionName] = {}
        NetworkDict[CollectionName]['Matrix'] = np.frombuffer(ConnectedMatrix[0]['ConnectedMatrix']['bytes'],
                                                    dtypeList[ConnectedMatrix[0]['ConnectedMatrix']['dtype']]).\
                                                    reshape(ConnectedMatrix[0]['ConnectedMatrix']['shape'])
                                                    
        NetworkDict[CollectionName]['GroupIndexList'] = ConnectedMatrix[0]['GroupIndexList']
