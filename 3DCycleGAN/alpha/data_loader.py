#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:19:43 2021

@author: arun
"""
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import h5py


def dataload3D_2(DataPath,patch_size,depth_size):
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
    CTsiz1=CT.shape  
    CBCTCellRef=mat_contents['CBCTInfocell']
    CBCLen=np.shape(CBCTCellRef)
    # CBCTs=[]
    CBCTi=np.random.choice(CBCLen[1],replace=False)  
    CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
    CBCellRef=mat_contents[CBCellRef]
    CBCT=CBCellRef[()]
    CBCT=np.transpose(CBCT,(2,1,0))
    CBsiz=CBCT.shape
    # CTblocksN=(CTsiz1[0]*CTsiz1[1]*CTsiz1[2])//(self.patch_size*self.patch_size*4*self.depth_size)
    # CBblocksN=(CBsiz[0]*CBsiz[1]*CBsiz[2])//(self.patch_size*self.patch_size*4*self.depth_size)
    # i=512//2
    # j=512//2
    # zi=1
    i=np.random.randint(CTsiz1[0]-patch_size*2)
    j=np.random.randint(CTsiz1[1]-patch_size*2)
    zi1=np.random.randint(CTsiz1[2]-depth_size)
    zi2=np.random.randint(CBsiz[2]-depth_size)
    CTblocks=CT[i:i+patch_size*2,j:j+patch_size*2,zi1:zi1+depth_size]
    CBblocks=CBCT[i:i+patch_size*2,j:j+patch_size*2,zi2:zi2+depth_size]
    # CTblocks=CT[0:32,0:32, 0:32]
    # CBblocks=CBCT[0:32,0:32,0:32]
    CTblocks=np.expand_dims(CTblocks, axis=0)
    CBblocks=np.expand_dims(CBblocks, axis=0)

    yield tf.convert_to_tensor(CTblocks, dtype=tf.float32), tf.convert_to_tensor(CBblocks, dtype=tf.float32)