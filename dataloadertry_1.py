#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:39:24 2021

@author: arun
"""
import time
import datetime
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
# import scipy.io as sio
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()

#works for mat file version 7.3 which is the new default.

def dataload(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles.sort()
    matfiles=[join(mypath,f) for f in onlyfiles]
    mat_fname_ind=np.random.choice(len(matfiles),replace=False)
    # mat_fname_ind=7
    mat_contents=h5py.File(matfiles[mat_fname_ind])
    # mat_contents=h5py.File(mat_fname)
    mat_contents_list=list(mat_contents.keys())
    print(matfiles[mat_fname_ind])
    PlanCTCellRef=mat_contents['CTInfoCell']
    CTLen=np.shape(PlanCTCellRef)
    if CTLen[1]>1:
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
    else:
        CTindex=0
    PlanCTLocRef=mat_contents['CTInfoCell'][1, CTindex]
    PlanCTLocRef=mat_contents[PlanCTLocRef]
    PlanCTLoc=PlanCTLocRef.value
    PlanCTCellRef=mat_contents['CTInfoCell'][2, CTindex]
    PlanCTCellRef=mat_contents[PlanCTCellRef]
    CT=PlanCTCellRef.value
    CT=np.transpose(CT,(2,1,0))
    CTsiz1=CT.shape
    CT_rand_index=np.random.choice(CTsiz1[2],size=batch_size,replace=False)
    batch_CT_img=np.zeros((CTsiz1[0],CTsiz1[1],len(CT_rand_index)))
    for ri in range(len(CT_rand_index)):
        batch_CT_img[:,:,ri]=CT[:,:,CT_rand_index[ri]]
    
    CBCTCellRef=mat_contents['CBCTInfocell']
    CBCLen=np.shape(CBCTCellRef)
    #Sequential CBCT scan selection
    CBCTs=[]
    for CBCTi in range(CBCLen[1]):
        # print(CBCTi)
        CBCellRef=mat_contents['CBCTInfocell'][2, CBCTi]
        CBCellRef=mat_contents[CBCellRef]
        CBCT=CBCellRef.value
        CBCT=np.transpose(CBCT,(2,1,0))
        CBCTs.append(CBCT)
        CBLocRef=mat_contents['CBCTInfocell'][1, CBCTi]
        CBLocRef=mat_contents[CBLocRef]
        CBCTLoc=CBLocRef.value
    CBsiz=CBCT.shape
    batch_CB_img=np.zeros((CTsiz1[0],CTsiz1[1],batch_size))
    for cbi in range(batch_size):
        CB_rand_sl_index=np.random.choice(CBsiz[2])
        CB_rand_pat_index=np.random.choice(CBCLen[1],replace=False)
        # print(CB_rand_pat_index)
        # print(CB_rand_sl_index)
        batch_CB_img[:,:,cbi]=CBCTs[CB_rand_pat_index][:,:,CB_rand_sl_index]
    return CT, CBCTs, batch_CT_img, batch_CB_img
#%%
# mat_fname='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB3/ZC003-DB3.mat'
mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB3/'

# print(mat_fname)
batch_size=100
CT,CBCTs,batch_CT_img, batch_CB_img=dataload(mypath)  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/1
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s sec'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)