#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:04:01 2021

@author: cuilab
"""

from MongoNeo.MongoReader import MongoReadModule
import pymongo
import quantities as pq
from elephant.kernels import GaussianKernel
from mpi4py import MPI 
import gc

def MPIMongoNeo(CollectionList):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    numjobs=len(CollectionList)
    
    njob_per_worker = int(numjobs/size)
    # the number of jobs should be a multiple of the NumProcess[MPI]
    
    this_worker_job = [CollectionList[x] for x in range(rank*njob_per_worker, (rank+1)*njob_per_worker) if x<numjobs]
    
    # map the index to parameterset [eps,anis]
    for CollectionName in this_worker_job:

        MongoReader = MongoReadModule(collection = CollectionName,
                                          SegName = 'FinalEnsemble_spike_train',
                                          Saverip="mongodb://10.10.44.85:27017/",
                                          db='Caesar_monkey',
                                          username='yongxiang',
                                          password='cuilab322')
            
        if '3T' in CollectionName:
            aligned_marker = 6
        else:
            aligned_marker = 5
    
        MongoReader.SpikeFiringRate(Statistics = 'instantaneous_rate',
                                    sampling_period = 1*pq.ms, 
                                    t_left = -1200*pq.ms, 
                                    t_right = 1200*pq.ms,
                                    kernel = GaussianKernel(3.66*pq.ms),
                                    aligned_marker = aligned_marker,
                                    TrialError=0,
                                    StatisticsML = None,
                                    DeleteQuery=False)
        gc.collect()

if __name__ == '__main__':
    myclient = pymongo.MongoClient("mongodb://10.10.44.85:27017/", username='admin', password='cuilab324')
    mydb = myclient['Caesar_monkey']
    CollectionList = mydb.list_collection_names()
    MPIMongoNeo(CollectionList)