#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 22:03:01 2021

@author: cuilab
"""
import numpy as np

def GenerateMlDict(mldata):
    mldata_dict = {'name':'mldata','mldata':{}}
    mldata_dict['mldata']['Trial'] = [i[0][0].astype(float) for i in mldata['Trial']]
    mldata_dict['mldata']['Block'] = [i[0][0].astype(float) for i in mldata['Block']]
    mldata_dict['mldata']['Condition'] = [i[0][0].astype(float) for i in mldata['Condition']]
    mldata_dict['mldata']['AbsoluteTrialStartTime'] = [i[0][0].astype(float) for i in mldata['AbsoluteTrialStartTime']]
    mldata_dict['mldata']['TrialError'] = [i[0][0].astype(float) for i in mldata['TrialError']]
    mldata_dict['mldata']['BehavioralCodes'] = {}
    mldata_dict['mldata']['BehavioralCodes']['CodeTimes'] = [list(i['CodeTimes'][0][0].squeeze().astype(float)) for i in mldata['BehavioralCodes']]
    mldata_dict['mldata']['BehavioralCodes']['CodeNumbers'] = [list(i['CodeNumbers'][0][0].squeeze().astype(float)) for i in mldata['BehavioralCodes']]
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
    return mldata_dict