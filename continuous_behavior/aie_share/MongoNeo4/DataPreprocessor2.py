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
import scipy.stats
import scipy.signal
from tqdm import tqdm
import multiprocessing
from .MongoNeoInterface import MongoNeoInterface
from .BaseInterfaceTools import VariableDescriptionInterface,StatisticsResults2SpikeData4DP,SearchAndUpload
from numba import jit
from numba.typed import List
from affinewarp import PiecewiseWarping
from difflib import SequenceMatcher

try:
    multiprocessing.set_start_method('spawn')
except:
    pass

def SliceFun(j,irrSliceResults,kargList):
    '''
    Slicing neo continuous data

    Parameters
    ----------
    j : neo list obj
        continuous data.
    irrSliceResults : dict
        saving slicing result.
    kargList : list
        slicing kargs.

    Returns
    -------
    None.

    '''
    irrSliceResults[j.name] = []
    for i in kargList:
        irrSliceResults[j.name].append(j.time_slice(i['t_start'],i['t_stop']))

@jit(nopython=True)
def spike_time(NumbaSpikeTrains,SlicedNumbaSpikeTrains,neurons,
               temp_array, t_start, t_stop):
    neuronInd = 0
    for st in NumbaSpikeTrains:
        left = 0
        right = st.shape[0]-1
        t_start_index = -1
        t_stop_index = -1
        while left<=right:
            center = (left+right)//2
            if st[center]==t_start: 
                t_start_index = center
                break
            elif st[center]>t_start:
                if st[center-1]<t_start:
                    t_start_index = center
                    break
                else:
                    right=center-1
            elif st[center]<t_start:
                left=center+1
                
        left = t_start_index
        right = st.shape[0]-1
        
        if t_start_index!=-1:
            while left<=right:
                center = (left+right)//2
                if st[center]==t_stop: 
                    t_stop_index = center
                    break
                elif st[center]>t_stop:
                    right=center-1
                elif st[center]<t_stop:
                    if st[center+1]>t_stop:
                        t_stop_index = center
                        break
                    else:
                        left=center+1
                        
        if t_start_index == -1 or t_stop_index == -1: 
            neuronInd = neuronInd+1
            SlicedNumbaSpikeTrains.append(SlicedNumbaSpikeTrains[0])
            continue
        
        if not (st[t_start_index]>=t_start and st[t_start_index-1]<t_start): raise TypeError
        if not (st[t_stop_index]<=t_stop and st[t_stop_index+1]>t_start): raise TypeError
        Slicedst = st[t_start_index:t_stop_index]
        SlicedNumbaSpikeTrains.append(Slicedst-t_start)
        temp_array[0:len(Slicedst)] = neuronInd
        neurons.append(temp_array[0:len(Slicedst)].copy())
        neuronInd = neuronInd+1

    return SlicedNumbaSpikeTrains

def time_histogram(spiketrains,data,kargsList):

    if 'binsize' in kargsList[0]: binsize = kargsList[0]['binsize'].rescale(spiketrains[0].units)   
    
    t_start = kargsList[0]['t_start'].rescale(binsize.units)
    t_stop = kargsList[0]['t_stop'].rescale(binsize.units)
    
    t_duration = float(round(float((t_stop-t_start).magnitude),
                                 int((binsize.units/pq.ms+3).magnitude)))
    FloatBin = float(binsize.magnitude)
    binnum = round(t_duration/FloatBin,1)
    if 'binnum' in kargsList[0]: binnum = kargsList[0]['binnum']
    binned = data.bin_spikes(int(binnum)).swapaxes(1, -1)
    
    return binned

def rescaled_time_histogram(data,binnum,kargsList):
    binned = []
    for ind in range(max(data.trials)):
        SelectedData = data.select_trials([ind])
        SelectedData.trials[:] = 0
        SelectedData.n_trials = 1
        SelectedData.tmax = (kargsList[ind]['t_stop']-kargsList[ind]['t_start']).rescale(pq.ms).magnitude.squeeze()
        binned.append(SelectedData.bin_spikes(binnum).squeeze().T)
    return np.array(binned)

def instantaneous_rate(TimeHistogram,kargsList,cutoff=3.0):
     
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
    

    TimeHistogram = np.concatenate((TimeHistogram,np.zeros(TimeHistogram.shape[0:2])[:,:,np.newaxis]),2)
    ConvolveKernel = kernel(t_arr).rescale(pq.Hz).magnitude[np.newaxis,:].repeat(TimeHistogram.shape[1],0)
    ConvolveKernel = ConvolveKernel[np.newaxis,:,:].repeat(TimeHistogram.shape[0],0)
    InstantaneousRate = scipy.signal.fftconvolve(TimeHistogram,
                                                 ConvolveKernel,
                                                 'same',
                                                 axes = 2)[:,:,0:-1]   
    
    return InstantaneousRate

def spiketrain_AffineWarp(data,kargsList,spiketrains):
    BINSIZE = kargsList[0]['binsize'].rescale(pq.ms).magnitude   # ms
    NBINS = int((data.tmax - data.tmin) / BINSIZE)
    MAXLAG = 0.1
    binned = data.bin_spikes(NBINS)
    # Create model.
    model = PiecewiseWarping(maxlag=MAXLAG)
    WarpResult = {}
    tmax = data.tmax
    trial_range = np.arange(len(binned))
    WarpResult['annotation'] = {}
    if kargsList[0]['aligned_marker']==kargsList[0]['aligned_marker2']:
        t0 = np.array([-i['t_left'] for i in kargsList])
        model.manual_fit(t0,binned, recenter=True)
        warped_data = model.transform(data)
        WarpResult['annotation']['warped_event'] = model.event_transform(trial_range, t0 / tmax) * tmax
        
    else:
        if kargsList[0]['0_point']=='t_start':
            t0 = np.array([-i['t_left'] for i in kargsList])
            t1 = np.array([i['DiffTime']-i['t_right'] for i in kargsList])
        else:
            t0 = np.array([tmax-i['DiffTime']-i['t_left'] for i in kargsList])
            t1 = np.array([tmax-i['t_right'] for i in kargsList])
            
        model.manual_fit(binned, t0, t1, recenter=False)
        warped_data = model.transform(data)
        WarpResult['annotation']['warped_event'] = np.array([model.event_transform(trial_range, t0 / tmax) * tmax,
                                                             model.event_transform(trial_range, t1 / tmax) * tmax])
    WarpResult['SpikeData'] = np.array([warped_data.trials,
                                        warped_data.spiketimes,
                                        warped_data.neurons])
    
    return WarpResult


def elephant_preprocessing(Seg,kargs,br_gain,kargsList):
    kargs = kargs.copy()      
    kargs['aligned_marker'] = kargs['aligned_marker'] if isinstance(kargs['aligned_marker'],list) else [kargs['aligned_marker']]
    kargs['aligned_marker2'] = kargs['aligned_marker2'] if isinstance(kargs['aligned_marker2'],list) else [kargs['aligned_marker2']]
    
    CodeIndex = np.array([i in kargs['aligned_marker'] for i in Seg.events[0].labels])
    CodeIndex2 = np.array([i in kargs['aligned_marker2'] for i in Seg.events[0].labels])
    #CodeIndex2 = Seg.events[0].labels==kargs['aligned_marker2']
    if (Seg.description['TrialError']==kargs['TrialError']) and (sum(CodeIndex)!=0) and (sum(CodeIndex2)!=0): 
        for i,j in zip(Seg.events[0].times[CodeIndex],Seg.events[0].times[CodeIndex2]):
            CodeTime = Seg.events[0].times[CodeIndex]+Seg.description['AbsoluteTrialStartTime']+br_gain+kargs['t_left']
            CodeTime2 = Seg.events[0].times[CodeIndex2]+Seg.description['AbsoluteTrialStartTime']+br_gain+kargs['t_right']
            kargs.update({'t_start':CodeTime,'t_stop':CodeTime2})   
            kargsList.append((kargs.copy(),Seg))

class DataPreprocessor(MongoNeoInterface):
    def __init__(self,
                 collection,
                 SegName,
                 Saverip="mongodb://localhost:27017/",
                 db="Poisson_monkey_array",
                 username='admin',
                 password='cuilab324',
                 mlDataIndex=1,
                 LFP=False,
                 NotLoadIndexList = ['analogsignals']):
        '''
        Datapreprocessor layer kernal 
        Integrating neural data & behavior data according to meta data 'neo'\
            (this 'ability' is inherited from MongoNeoInterface)
        Providing neural data (spike trains) statistics method
        
        Parameters
        ----------
        collection : str
            mongodb collection name, representing a set in one animal experiment.
        SegName : str
            spike trains segment name, the spike trains was managed to 'segments' of neo \
                if can be defined by sorter's name or other potential nickname, depends on researcher.
        Saverip : str, optional
            mongodb ip. The default is "mongodb://localhost:27017/".
        db : str, optional
            mongodb db, representing the experimental animals name and recording method. The default is "Poisson_monkey_array".
        username : str, optional
            mongodb username, representing the data analyzer (should be). The default is 'admin'.
        password : str, optional
            mongodb user password, for mongodb login. The default is 'cuilab324'.
        mlDataIndex : TYPE, optional
            Behavior data allocation index. The default is 1.
        LFP : bool, optional
            Option for LFP data searching (because it is too large!). The default is False.
        NotLoadIndexList : list, optional
            list of behavior data attribution name, the names in this list are not searching from the database\
                (for accelerating). The default is ['analogsignals'].

        Returns
        -------
        The instance of DataPreprocessor.

        '''
        
        #running MongoNeoInterface construction function to searching mongoneo data
        super().__init__(collection,dbName=db,
                MongoAdress=Saverip,
                username=username, password=password)
        # set attribution of the instance
        self.username = username
        self.vdi = VariableDescriptionInterface() #this is a mongo interface module for data transformation
        self.block_br = self._Decoder(SegName,0)
        self.mldata = self._Decoder(SegName,mlDataIndex,NotLoadIndexList)
        # select spike train related to SegName and set their t_stop (used for spike train slicing, seemed useless in lateset version)
        self.SegName = SegName
        # self.spiketrains = [i.spiketrains for i in self.block_br.segments if i.name in SegName][0]

        # for i in range(len(self.spiketrains)):
            # self.spiketrains[i].t_stop = max(self.spiketrains[i][-1],self.block_br.segments[0].events[0].times[-1])
        # aligning behavior marker time to recording system marker time
        BlockEvent = [i for i in self.block_br.segments if i.name in 'RecordingSystemEvent'][0].events[0]
        MlSeg = self.mldata.segments[0]
        MlEvent = MlSeg.events[0]

        MlEventMarker = MlEvent.labels
        NevEventMarker = np.array([float(i) for i in BlockEvent.labels[0:len(MlEvent.labels)]])
    
        
        if sum(MlEventMarker==NevEventMarker)==len(MlEventMarker):
            self.gain = BlockEvent[0]-(MlEvent[0]+MlSeg.description['AbsoluteTrialStartTime'])
        else:
            print('Wrong marker aligning, try to align whole event marker')
            AllMlEvent = np.concatenate([i.events[0].labels for i in self.mldata.segments]).astype(int)
            AllMlEventTime = np.concatenate([i.events[0].times+i.description['AbsoluteTrialStartTime'] for i in self.mldata.segments])
            BlockEventLabel = BlockEvent.labels.astype(int)            
            
            if len(BlockEventLabel)>len(AllMlEvent):
                BlockEventLabel = BlockEventLabel[0:len(AllMlEvent)]
            
            Interval = round(len(BlockEventLabel)/2)
            AlignIndex=-1
            for i in range(Interval):
                AlignResults = AllMlEvent[i:i+Interval]==BlockEventLabel[0:Interval]
                if sum(AlignResults)/Interval>0.9:
                    for j in range(int(AlignResults.shape[0]/2)):
                        if sum(AlignResults[j:j+200])==200:
                            AlignIndex = i+j
                            MLAlignIndex = i+j
                            RLAlignIndex = j
                            print('Successed')
                            break
                    break            
                         
            if AlignIndex<0:
                print("Wrong marker aligning, please give a 'gain'")
            else:
                self.gain = BlockEvent[RLAlignIndex:RLAlignIndex+200].times-AllMlEventTime[MLAlignIndex:MLAlignIndex+200]*self.mldata.segments[0].events[0].times.units
                self.gain = self.gain.mean()
        # save spike trains (units) name, seemed useless in lateset version)
        #self.SpikeTrainName = [sts.description['ElectrodeLabel']+' clu'+ str(int(sts.description['clu'])) for sts in self.spiketrains]
        """try:
            #self.SpikeTrainName = [sts.description['ElectrodeLabel']+' clu'+ str(sts.description['clu']) for sts in self.spiketrains]
        except KeyError:
            pass"""
        # create new mongodb collection for avoiding index conflict to experiment data collection
        self.mycol = self.mydb[collection+'_DP']
        
    def SpikeStatistics(self,**kargs):
        '''
        Method for spike statistics
        three statistics method providing, spike time slicing, time histogram, instantaneous rate
        the time_warping option of spike time slicing can be set True and the spike slicing result\
            will be sliced then warped according to ganguli's work

        Parameters
        ----------
        **kargs : dict
            input parameters for getting appropriate statistics results.
        kargs containing the following parameters:
            'Statistics' : str
                {'spike_time', 'time_histogram', 'instantaneous_rate'}, spike statistics method\
                    related to spike time slicing, time histogram and instantaneous rate
            't_left': float*quantities_units
                left boundary of time window (start from aligned_marker)
            't_right': float*quantities_units
                right boundary of time window (end from aligned_marker2)
            'aligned_marker' : float 
                define time window left boundary according to event marker
            'aligned_marker2' : float 
                define time window right boundary according to event marker, optional\
                    if only set 'aligned_marker', the 'aligned_marker2' will set same as 'aligned_marker'
            'TrialError' : float 
                which kind of error of trials you want to select
            'StatisticsML' : list
                Selected trial for analysis, optional
            'time_warping' : bool 
                whether the sliced time data needed to be warped, optional
            'binnum' : float 
                if you give time window with different width and not set 'aligned_1st_marker',\
                    should tell this module how many bin number you would like to take, optional
            'aligned_1st_marker' : bool
                spike statistics needs time window with same length, if the time length of all trials \
                    are not the same, the module will expand the time window aligned to 1st or 2nd marker, according to\
                        this option.

        Returns
        -------
        list or structure
            if 'Statistics' is not 'spike_time', the return is a list containing all statistics result related to\
                the trial, which set as StatisticsMList attribution.
            if 'Statistics' is 'spike_time', the return is a SpikeData4DP structure and its trial is set as StatisticsMList attribution.         

        '''
        
        # set InsertDict as label of statistics results
        if 'aligned_marker2' not in kargs.keys(): kargs['aligned_marker2']=kargs['aligned_marker'] # if not set 'aligned_marker2', this line will be done
        self.kargs = kargs # this should not be public
        self.InsertDict = kargs.copy()
        self.InsertDict['collection'] = self.collection
        self.InsertDict['SegName'] = self.SegName
        self.InsertDict['name'] = 'Statistics'
        self._warpOption = False
        
        # define time_warping option and set _warpOption for determine of whether perform warping
        if 'time_warping' in self.InsertDict.keys(): 
            self._warpOption = self.InsertDict['time_warping']
            del self.InsertDict['time_warping']
        
        # avoid conflict of label
        if 'StatisticsML' in kargs.keys(): del self.InsertDict['StatisticsML']
        
        # get statistics parameters for each trial
        StatisticsFile = []
        for i in self.mldata.segments: elephant_preprocessing(i,kargs,self.gain,StatisticsFile)
        self.StatisticsMList = [x[1] for x in StatisticsFile if x != None]
        self.kargsList = [x[0] for x in StatisticsFile if x != None]
        self.InsertDict['username'] = self.username # give username label, seemed not be set as public
        
        # different way to handle 2 marker
        if kargs['aligned_marker']!=kargs['aligned_marker2']:
            TimeMax = max([i['t_stop']-i['t_start'] for i in self.kargsList])
            # to avoid the TimeMax can't devide 'binsize' or 'sampling_period' to integer
            if 'binnum' not in self.kargsList[0] and 'spike_time' not in kargs['Statistics']: 
                TimeMax = np.ceil(TimeMax/self.kargsList[0]['binsize']+1)*self.kargsList[0]['binsize'] if 'binsize' in self.kargsList[0]\
                    else np.ceil(TimeMax/self.kargsList[0]['sampling_period']+1)*self.kargsList[0]['sampling_period']
            # 'aligned_1st_marker' option branch, if not set this option, do nothing
            if 'aligned_1st_marker' in kargs:
                if kargs['aligned_1st_marker']:
                    for i in range(len(self.kargsList)):
                        DiffTime = self.kargsList[i]['t_stop']-self.kargsList[i]['t_start']
                        if DiffTime < TimeMax:
                            self.kargsList[i]['t_stop'] = self.kargsList[i]['t_stop']+TimeMax-DiffTime
                            self.kargsList[i]['DiffTime'] = DiffTime
                            self.kargsList[i]['0_point'] = 't_start'
                else:
                    for i in range(len(self.kargsList)):
                        DiffTime = self.kargsList[i]['t_stop']-self.kargsList[i]['t_start']
                        if DiffTime < TimeMax:
                            self.kargsList[i]['t_start'] = self.kargsList[i]['t_start']-TimeMax+DiffTime
                            self.kargsList[i]['DiffTime'] = DiffTime
                            self.kargsList[i]['0_point'] = 't_stop'
        
        # select statistics method
        if kargs['Statistics'] == 'time_histogram':
            StatisticList = self._GetTimeHistogram()
        if kargs['Statistics'] == 'instantaneous_rate':
            StatisticList = self._GetInstantaneousRate()
        if kargs['Statistics'] == 'spike_time':
            StatisticList = self._GetSpikeTime()
        
        # if 'StatisticsML' is set, seemed the user need trial selection, this part will operate
        if 'StatisticsML' in self.kargs.keys():
            # spike time structure is not like other, so needs a branch
            if 'spike_time' not in kargs['Statistics']:             
                SelectedIndex = [ind for ind,i in enumerate(self.StatisticsMList) if i.index in self.kargs['StatisticsML']]   
                StatisticList = StatisticList[np.array(SelectedIndex)]
                self.StatisticsMList = [i for i in self.StatisticsMList if i.index in self.kargs['StatisticsML']]
            
            else:
                
                self.StatisticsMList = [i for i in self.StatisticsMList if i.index in self.kargs['StatisticsML']]
                ArrayIndex = StatisticList.annotation['mlIndex']
                StatisticList = StatisticList.select_trials(np.array([np.where(ArrayIndex==i)[0][0]\
                                                            for i in self.kargs['StatisticsML'] if i in ArrayIndex]),
                                                                    inplace = True)        
        return StatisticList
    
    def _GetTimeWarping(self):
        '''
        Private method for time warping function
        this method is used to warp spike time by piecewise warp (manual_fit)
        first this function search paired karg in the database, if the searching result is none\
            the function will perform related analysis method and send it to database

        Returns
        -------
        structure
            instance of inherited SpikeData.

        '''
        
        # change kargs to insert label
        InsertDict = self.InsertDict.copy()
        InsertDict['Statistics'] = 'warped_spike_time'
        if 'binsize' in InsertDict.keys(): del InsertDict['binsize']
        if 'sampling_period' in InsertDict.keys(): del InsertDict['sampling_period']
        if 'kernel' in InsertDict.keys(): del InsertDict['kernel']
        
        # analysis performing method
        @SearchAndUpload
        def _FindSpikeStatistics(self):
            data = self._GetSpikeTime()
            SpikeDataDict = spiketrain_AffineWarp(data,self.kargsList,self.spiketrains)
            SpikeDataDict['annotation']['mlIndex'] = np.array([i.index for i in self.StatisticsMList]) # save related trials
            return SpikeDataDict
        StatisticsResults = _FindSpikeStatistics(self,InsertDict)
        # use sliced data to construct modified Spikedata structure
        data = StatisticsResults2SpikeData4DP(StatisticsResults,self.spiketrains,self.kargsList)
        return data
        
    def _GetSpikeTime(self):
        '''
        Use binary search for spike train slicing

        Returns
        -------
        structure
            instance of inherited SpikeData..

        '''
        # change kargs to insert label
        InsertDict = self.InsertDict.copy()
        InsertDict['Statistics'] = 'spike_time'
        if 'binsize' in InsertDict.keys(): del InsertDict['binsize']
        if 'sampling_period' in InsertDict.keys(): del InsertDict['sampling_period']
        if 'kernel' in InsertDict.keys(): del InsertDict['kernel']
        # analysis performing method, use numba for acceleration
        @SearchAndUpload
        def _FindSpikeStatistics(self):
            # build numba list as a container of spike trains with different length
            NumbaSpikeTrains = List()
            for st in self.spiketrains:
                NumbaSpikeTrains.append(np.array(st.magnitude))
                
            # build numba list as a container of slicing result
            SlicedNumbaSpikeTrains = List()
            neurons = List()
            # list saver
            SpikeTime = []
            trials = []
            # temp vector for neuron index saving
            MaxTime = int(max([i['t_stop']-i['t_start'] for i in self.kargsList]).rescale(pq.ms).magnitude)*10
            temp_array = np.zeros((MaxTime, ))
            neurons.append(temp_array)
            # slicing
            for TrialInd, i in enumerate(self.kargsList):
                SlicedNumbaSpikeTrains.append(np.array([]))
                SlicedSpike = np.concatenate(spike_time(NumbaSpikeTrains,
                                                        SlicedNumbaSpikeTrains,
                                                        neurons,
                                                        temp_array,
                                                        float(i['t_start'].rescale(self.spiketrains[0].units).magnitude),
                                                        float(i['t_stop'].rescale(self.spiketrains[0].units).magnitude))[1::])
                SpikeTime.extend(SlicedSpike)
                trials.extend([TrialInd]*len(SlicedSpike))
                SlicedNumbaSpikeTrains.clear()
            # integrate slicing result as a dict
            SpikeDataDict = {}
            SpikeDataDict['SpikeData'] = np.array([np.array(trials).squeeze(),
                                                   np.array(SpikeTime).squeeze(),
                                                   np.concatenate(neurons[1::]).squeeze()])
            # add annotation key for adding other related infomation
            SpikeDataDict['annotation'] = {}
            SpikeDataDict['annotation']['mlIndex'] = np.array([i.index for i in self.StatisticsMList])
            SpikeDataDict['annotation']['units'] = self.spiketrains[0].units
            return SpikeDataDict
        
        # a branch of whether using warping, return SpikeData structure
        if self._warpOption:
            return self._GetTimeWarping()
        else:        
            StatisticsResults = _FindSpikeStatistics(self,InsertDict)
            self.debuger = StatisticsResults
            data = StatisticsResults2SpikeData4DP(StatisticsResults,self.spiketrains,self.kargsList)
            return data   
    
    def _GetTimeHistogram(self):
        '''
        Get time histogram from sliced spike train, this ability is derived from SpikeData structure\
            _GetSpikeTime function will first perform to get sliced spike train, then performing getting time histogram

        Returns
        -------
        np.ndarray
            a spike count array each dimention related to trial index, cell index and time.

        '''
        # change kargs to insert label
        InsertDict = self.InsertDict.copy()
        InsertDict['Statistics'] = 'time_histogram'
        if 'sampling_period' in InsertDict.keys(): del InsertDict['sampling_period']
        if 'kernel' in InsertDict.keys(): del InsertDict['kernel']
        if 'binsize' not in InsertDict.keys(): InsertDict['binsize'] = self.InsertDict['sampling_period']
        # analysis performing method, use numba for acceleration (provided by SpikeData structure)
        @SearchAndUpload
        def _FindSpikeStatistics(self):
            # because not sure about whether time length is same in each spike trains, this part gives a branch
            if self.kargsList[0]['aligned_marker']!=self.kargsList[0]['aligned_marker2']\
                and 'aligned_1st_marker' not in self.kargsList[0]:
                # if the condition is true, indicating the time length must be different in each trial, so must give an appointed 'binnum'
                assert 'binnum' in self.kargsList[0], "Please send me a 'binnum'" #use assert for 'binnum' appointment
                # use anothor function for using same binnum in different trial data preprocessing 
                TimeHistogram = rescaled_time_histogram(self._GetSpikeTime(),self.kargsList[0]['binnum'],self.kargsList)
            else:
                # normal preprocessed time histogram
                TimeHistogram = time_histogram(self.spiketrains,self._GetSpikeTime(),self.kargsList)
            return TimeHistogram
        # return time histogram array
        return _FindSpikeStatistics(self,InsertDict)
    
    def _GetInstantaneousRate(self):
        '''
        Get instantaneous rate from time histogram.
        Using fft for convolution, the convolution parameters are derived from elephant

        Returns
        -------
        np.ndarray
            a instantaneous rate array each dimention related to trial index, cell index and time.

        '''
        # change kargs to insert label
        InsertDict = self.InsertDict.copy()
        InsertDict['Statistics'] = 'instantaneous_rate'
        if 'binsize' in InsertDict.keys(): del InsertDict['binsize']
        # analysis performing method, using scipy, might be further accelerated by cusignal
        @SearchAndUpload
        def _FindSpikeStatistics(self):        
            # Because the  _GetTimeHistogram need binsize input,get it from sampling_period
            self.kargsList[0]['binsize'] = self.kargs['sampling_period']
            InstantaneousRate = instantaneous_rate(self._GetTimeHistogram(),
                                                   self.kargsList,cutoff=3.0)
            return InstantaneousRate
        # return instantaneous rate array
        return _FindSpikeStatistics(self,InsertDict)
    
    def GetSlice(self,SubSegData,**kargs):
        '''
        Slicing continuous data

        Parameters
        ----------
        SubSegData : object
            Analog or irregularsample data, handled by neo.
        **kargs : dict
            same as kargs in SpikeStatistics, optional, if kargs is none,\
                it will use the parameter same as used in SpikeStatistics.

        Returns
        -------
        SliceResults : list
            sliced continuous data same index as input, sliced results are nested in this list.

        '''
        
        # a branch to find sliced parameter, if kargs is not none, using function elephant_preprocessing get trial related slicing parameters
        if len(kargs)!=0:
            # same as SpikeStatistics parameters preprocessing
            br_gain = self.gain
            StatisticsFile = []
            for i in self.mldata: elephant_preprocessing(i,kargs,br_gain,StatisticsFile)
            self.kargsList = [x[0] for x in StatisticsFile if x != None] # trial related slicing parameters
        
        # set container
        SliceResults = [[] for _ in SubSegData]
        for ind, j in enumerate(SubSegData):
            for i in tqdm(self.kargsList):
                # slicing method is derived from neo data object
                SliceResults[ind].append(j.time_slice(i['t_start'],i['t_stop']))
            
        return SliceResults
    
    def _ReadOut(self,InsertDict): 
        '''
        Searching data in mongodb according to parameters

        Parameters
        ----------
        InsertDict : dict
            parameters of data analyzing, used as searching keys.

        Returns
        -------
        dict
            data & parameters in mongodb.

        '''
        # find keys in mongodb, every analysis results will be handed as a query
        DbReadOut = self.mycol.find_one(self.vdi.encode(InsertDict),{'_id': 0})
        # transform mongodb query to python data structure as return
        return self.vdi.decode(DbReadOut,self.fs) if isinstance(DbReadOut,dict) else None
    
    def _Upload(self,InsertDict,Input):
        '''
        integrate parameters and analysis results, then upload to mongodb

        Parameters
        ----------
        InsertDict : dict
            kargs containing analysis parameters.
        Input : object
            analysis results, in this function, the results are paired to 'UploadData'.

        Returns
        -------
        None.

        '''
        # integrate data
        SubInsertDict = InsertDict.copy()
        SubInsertDict['UploadData'] = Input
        # transform python data to mongodb accepted data
        EncodedInsertDict = self.vdi.encode(SubInsertDict,self.fs)
        # insert it to mongodb, the collection is defined at the instance construction
        self.mycol.insert_one(EncodedInsertDict)
        