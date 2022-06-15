#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:04:01 2021

@author: cuilab
"""

import pymongo
import quantities as pq
from mpi4py import MPI 
import gc
from MongoNeo.RepresentationAnalysis.DirectionalPD import DirectionalPDFinder
import numpy as np

def MPIMongoNeoPD(TimeStep,CollectionList):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    numjobs=len(CollectionList)
    
    njob_per_worker = int(numjobs/size)
    # the number of jobs should be a multiple of the NumProcess[MPI]
    
    this_worker_job = [CollectionList[x] for x in range(rank*njob_per_worker, (rank+1)*njob_per_worker) if x<numjobs]
    
    # map the index to parameterset [eps,anis]
    for CollectionName in this_worker_job:
        PDFinder = DirectionalPDFinder(collection = CollectionName,
                                      SegName = 'Kilo_spike_train',
                                      Saverip="mongodb://10.10.47.78:27017/",
                                      db='Laplace_monkey_Sprobe',
                                      username='yongxiang',
                                      password='cuilab322',
                                      TrialError=0,
                                      LFP=False)
        
        Epoch = 3
        Object = 4
        for step in TimeStep:
            
            UserDirection = [np.math.atan2(*tuple(np.array(i.irregularlysampledsignals[2])[Epoch,Object,::-1]))\
                         for i in PDFinder.StatisticsTrialList]
                
            UserDirection = list(np.round(np.array(UserDirection)/(np.pi/3))*np.pi/3)
            
            PDFinder.GetPDSpikeStatistics(Statistics = 'time_histogram',
                                        t_left = -1000*pq.ms+step*50*pq.ms, 
                                        t_right = -1000*pq.ms+step*50*pq.ms+50*pq.ms,
                                        aligned_marker = 5,
                                        cutoff = 3,
                                        FunName = 'DiscreatePD',
                                        UserDirection = UserDirection)
            PD = PDFinder.PD
        gc.collect()

if __name__ == '__main__':
    myclient = pymongo.MongoClient("mongodb://10.10.47.78:27017/", username='admin', password='cuilab324')
    mydb = myclient['Laplace_monkey_Sprobe']
    CollectionList = [i for i in mydb.list_collection_names() if '_PD' not in i or '.' not in i]
    MPIMongoNeoPD(list(range(40)),CollectionList)