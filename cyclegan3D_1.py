#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:18:53 2021

@author: arun
"""

import time
import datetime
import h5py
import numpy as np
from random import randint

from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
# import scipy.io as sio
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()

#works for mat file version 7.3 which is the new default.



DataPath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'

onlyfiles = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
onlyfiles.sort()
onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
onlyfiles = onlyfiles[0:-onlyfileslenrem]
matfiles=[join(DataPath,f) for f in onlyfiles]
mat_fname_ind=np.random.choice(len(matfiles),replace=False)

mat_contents=h5py.File(matfiles[mat_fname_ind])
mat_contents_list=list(mat_contents.keys())

PlanCTCellRef=mat_contents['CTInfoCell']
CTLen=np.shape(PlanCTCellRef)
CTsl=np.zeros([CTLen[1],1])
for cti in range(CTLen[1]):
    CTmatsizref=mat_contents['CTInfoCell'][1,cti]
    CTLocR=mat_contents[CTmatsizref]
    CTLoc=CTLocR.value
    CTsiz=np.shape(CTLoc)
    if CTsiz[1]>300:
        CTsl[cti]=1
    else:
        CTsl[cti]=0
CTindex=np.where(CTsl==1)
CTindex=CTindex[0]
CTindex=int(CTindex)
PlanCTLocRef=mat_contents['CTInfoCell'][1, CTindex]
PlanCTLocRef=mat_contents[PlanCTLocRef]
PlanCTLoc=PlanCTLocRef.value
PlanCTCellRef=mat_contents['CTInfoCell'][2, CTindex]
PlanCTCellRef=mat_contents[PlanCTCellRef]
PlanCT=PlanCTCellRef.value
PlanCT=np.transpose(PlanCT,(2,1,0))
batch_size=10
CTsiz1=PlanCT.shape
# CT_rand_index=np.random.choice(CTsiz1[2],size=batch_size,replace=False)
# batch_CT_img=np.zeros((CTsiz1[0],CTsiz1[1],len(CT_rand_index)))
# for ri in range(len(CT_rand_index)):
#     batch_CT_img[:,:,ri]=PlanCT[:,:,CT_rand_index[ri]]
PlanCTCellRef=mat_contents['CTInfoCell'][3, CTindex]
PlanCTCellRef=mat_contents[PlanCTCellRef]
PlanCTvoxel=PlanCTCellRef.value
CBCTCellRef=mat_contents['CBCTInfocell']
CBCLen=np.shape(CBCTCellRef)
#Random CBCT scan selection
CBCTi=randint(0,CBCLen[1]-1)
CBCellRef=mat_contents['CBCTInfocell'][2, CBCTi]
CBCellRef=mat_contents[CBCellRef]
CBCT=CBCellRef.value
CBCT=np.transpose(CBCT,(2,1,0))
CBLocRef=mat_contents['CBCTInfocell'][1, CBCTi]
CBLocRef=mat_contents[CBLocRef]
CBCTLoc=CBLocRef.value
#%%
#Sequential CBCT scan selection
# CBCTs=[]
# for CBCTi in range(CBCLen[1]):
#     # print(CBCTi)
#     CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
#     CBCellRef=mat_contents[CBCellRef]
#     CBCT=CBCellRef.value
#     CBCT=np.transpose(CBCT,(2,1,0))
#     CBCTs.append(CBCT)
#     CBLocRef=mat_contents['CBCTInfocell'][1, CBCTi]
#     CBLocRef=mat_contents[CBLocRef]
#     CBCTLoc=CBLocRef.value
# CBCellRef=mat_contents['CBCTInfocell'][3, CBCTi]
# CBCellRef=mat_contents[CBCellRef]
# CBCTvoxel=CBCellRef.value
# CBsiz=CBCT.shape
# # CB_rand_pat_index=np.random.choice(CBCLen[1],size=batch_size,replace=True)
# # batch_CB_img=np.zeros((CTsiz1[0],CTsiz1[1],len(CB_rand_pat_index)))
# batch_CB_img=np.zeros((CTsiz1[0],CTsiz1[1],batch_size))
# for cbi in range(batch_size):
#     CB_rand_sl_index=np.random.choice(CBsiz[2])
#     CB_rand_pat_index=np.random.choice(CBCLen[1],replace=False)
#     print(CB_rand_pat_index)
#     print(CB_rand_sl_index)
#     batch_CB_img[:,:,cbi]=CBCTs[CB_rand_pat_index][:,:,CB_rand_sl_index]


#%%  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)