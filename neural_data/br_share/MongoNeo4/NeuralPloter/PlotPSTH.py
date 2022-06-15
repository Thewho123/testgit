import numpy as np
from MongoNeo.MongoReader import MongoReadModule
      
class IRPloter(MongoReadModule):
    def __init__(self,
                 collection,
                 SegName,
                 Saverip="mongodb://localhost:27017/",
                 db="Caesar_monkey",
                 username='admin',
                 password='cuilab324'):
        super(IRPloter, self).__init__(collection,SegName,Saverip,
                                       db,username,password)
        self.ConditionIRDict = {}
        
    # Preprocessing method 
    def GetIR(self,**kargs):
        #%% Preprocessing
        assert 'instantaneous_rate' in kargs['Statistics'],'Please set Statistics to "instantaneous_rate"'
            
        FiringRate = self.SpikeFiringRate(Statistics = 'instantaneous_rate',
                                          sampling_period = kargs['sampling_period'], 
                                          t_left = kargs['t_left'], 
                                          t_right = kargs['t_right'],
                                          kernel = kargs['kernel'],
                                          aligned_marker = kargs['aligned_marker'],
                                          TrialError=kargs['TrialError'],
                                          StatisticsML = kargs['StatisticsML'])
        
        self.StatisticsML = [i['StatisticsML'] for i in self.spike_firing_rate]    
        FiringRate[FiringRate<0] = 0
        #%% Sent results to class property
        self.IRKargs = kargs.copy()
        IRBootsList = []
        for _ in range(kargs['bootstrap']):
            BootstrapIndex = np.random.randint(0,FiringRate.shape[0],(FiringRate.shape[0],))
            IRBootsList.append(FiringRate[BootstrapIndex].mean(0))
        self.IRBootsFiring = np.array(IRBootsList)
        self.ConditionIRDict[kargs['ConditionName']] = self.IRBootsFiring
        
    def GetStageObjectTrial(self,**kargs):
        ObjectPosition = []
        for i in self.StatisticsML:
            CuserObject = i['ObjectStatusRecord']['Position'][kargs['stage']][kargs['objectInd']]
            ObjectPosition = ObjectPosition+[CuserObject]
        return ObjectPosition
    
    def GetCondition(self,**kargs):
        Condition = [i['Condition'] for i in self.StatisticsML]
        return Condition
    
   
    
