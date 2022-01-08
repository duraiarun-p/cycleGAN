#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:50:41 2021

@author: arun
"""
import time
import datetime
import numpy as np
# from numpy.lib import stride_tricks
import h5py
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import tensorflow as tf

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
print('Script started at %s (Before functionality)'%st_0)
#%%
def dataload3D_2_predict(DataPath):
        # self.batch_size=1
        # mypath=self.DataPath
        onlyfiles = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
        onlyfiles.sort()
        onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
        onlyfiles = onlyfiles[0:-onlyfileslenrem]
        matfiles=[join(DataPath,f) for f in onlyfiles]
        mat_fname_ind=np.random.choice(len(matfiles),replace=False)  
        mat_contents=h5py.File(matfiles[mat_fname_ind],'r')
        # mat_contents_list=list(mat_contents.keys())    
        PlanCTCellRef=mat_contents['CTInfoCell']
        CTLen=np.shape(PlanCTCellRef)
        CTsl=np.zeros([CTLen[1],1])
        for cti in range(CTLen[1]):
            CTmatsizref=mat_contents['CTInfoCell'][1,cti]
            CTLocR=mat_contents[CTmatsizref]
            CTLoc=CTLocR[()]
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
        # PlanCTLoc=PlanCTLocRef[()]
        PlanCTCellRef=mat_contents['CTInfoCell'][2, CTindex]
        PlanCTCellRef=mat_contents[PlanCTCellRef]
        CT=PlanCTCellRef[()]
        CT=np.transpose(CT,(2,1,0))#data volume
        # CT=(CT-np.min(CT))/np.ptp(CT)
        CT = (CT-np.min(CT))/(np.max(CT)-np.min(CT))
        # CTsiz1=CT.shape  
        CBCTCellRef=mat_contents['CBCTInfocell']
        CBCLen=np.shape(CBCTCellRef)
        CBCTi=np.random.choice(CBCLen[1],replace=False)  
        CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
        CBCellRef=mat_contents[CBCellRef]
        CBCT=CBCellRef[()]
        CBCT=np.transpose(CBCT,(2,1,0))
        # CBCT=(CBCT-np.min(CBCT))/np.ptp(CBCT)
        CBCT=(CBCT-np.min(CBCT))/(np.max(CBCT)-np.min(CBCT))
        # CBsiz=CBCT.shape
        # i=np.random.randint(CTsiz1[0]-self.patch_size*2)
        # j=np.random.randint(CTsiz1[1]-self.patch_size*2)
        # zi1=np.random.randint(CTsiz1[2]-self.depth_size)
        # zi2=np.random.randint(CBsiz[2]-self.depth_size)
        # CTblocks=CT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi1:zi1+self.depth_size]
        # CBblocks=CBCT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi2:zi2+self.depth_size]
        # CTblocks=np.expand_dims(CTblocks, axis=0)
        # CBblocks=np.expand_dims(CBblocks, axis=0)
        return CT, CBCT
#%%
Datapath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
CT,CBCT=dataload3D_2_predict(Datapath)
CTsiz1=CT.shape 
CBsiz=CBCT.shape
patch_size=32
depth_size=32
threshold=1*np.mean(CT)
#%%
# data=CT
# blck=(32,32,32)
# strd=(16,16,16)
# sh = np.array(CTsiz1)
# blck = np.asanyarray(blck)
# strd = np.asanyarray(strd)
# nbl = (sh - blck) // strd + 1
# strides = np.r_[data.strides * strd, data.strides]
# dims = np.r_[nbl, blck]
# data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)

# for i in  range(0,CTsiz1[0],patch_size):
#     for j in range(0,CTsiz1[1],patch_size):
#         for zi in range(0,CTsiz1[2],depth_size):
#             # currentBlk=CT[i:i+patch_size,j:j+patch_size,zi:zi+depth_size]
#             # # currentBlk=currentBlk[:,:,0:depth_size]
#             # blksiz=currentBlk.shape
#             # if blksiz[2] != depth_size:
#             #     diff_zi=blksiz[2]-depth_size
#             #     zi=zi+diff_zi
#             #     currentBlk=CT[i:i+patch_size,j:j+patch_size,zi:zi+depth_size]
            
#             print('i=%s j=%s zi=%s'%(i,j,zi))
# from patchify import patchify, unpatchify
# patches = patchify(CT, (patch_size,patch_size,depth_size), step=16) # patch shape [2,2,3]

# CT_u = unpatchify(patches, CTsiz1)
CT_P=np.zeros_like(CT,dtype=float)
for i in  range(0,CTsiz1[0],patch_size):
    for j in range(0,CTsiz1[1],patch_size):
        for zi in range(0,CTsiz1[2],depth_size):
            currentBlk=CT[i:i+patch_size,j:j+patch_size,zi:zi+depth_size]
            # currentBlk=currentBlk[:,:,0:depth_size]
            blksiz=currentBlk.shape
            if blksiz[2] != depth_size:
                diff_zi=blksiz[2]-depth_size
                zi=zi+diff_zi
                currentBlk=CT[i:i+patch_size,j:j+patch_size,zi:zi+depth_size]
            currentBlk_i=np.expand_dims(currentBlk, axis=-1)
            currentBlk_i=np.expand_dims(currentBlk_i, axis=0)
            currentBlk_t=tf.convert_to_tensor(currentBlk_i, dtype=tf.float32)
            # currentBlk_p=tf.where(currentBlk_t > threshold, 1, 0)
            # currentBlk_p=tf.where(currentBlk_t > tf.reduce_mean(currentBlk_t), 1, 0)
            currentBlk_p = currentBlk_t.numpy()/2
            currentBlk_p = np.squeeze(currentBlk_p,axis=-1)
            currentBlk_p = np.squeeze(currentBlk_p,axis=0)
            CT_P[i:i+patch_size,j:j+patch_size,zi:zi+depth_size]=currentBlk_p

#%%
slice_index=np.random.choice(CTsiz1[2],replace=False)             
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(CT[:,:,slice_index],cmap='gray')
plt.show()
plt.show()
plt.title('CT')
plt.subplot(1,2,2)
plt.imshow(CT_P[:,:,slice_index],cmap='gray')
plt.show()
plt.show()
plt.title('pseudo CB')

#%%  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)