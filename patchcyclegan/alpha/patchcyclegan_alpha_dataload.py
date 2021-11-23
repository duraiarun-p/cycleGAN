#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:39:44 2021

@author: arun
"""
# import time
# import datetime
# import cv2
# import itk
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
import os

#%%
mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
depth_size=32
patch_size=16

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()
onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
onlyfiles = onlyfiles[0:-onlyfileslenrem]
matfiles=[join(mypath,f) for f in onlyfiles]
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
CBCTs=[]
CBCTi=np.random.choice(CBCLen[1],replace=False)  

CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
CBCellRef=mat_contents[CBCellRef]
CBCT=CBCellRef[()]
CBCT=np.transpose(CBCT,(2,1,0))

CBsiz=CBCT.shape

CTblocksN=(CTsiz1[0]*CTsiz1[1]*CTsiz1[2])//(patch_size*patch_size*4*depth_size)
CBblocksN=(CBsiz[0]*CBsiz[1]*CBsiz[2])//(patch_size*patch_size*4*depth_size)

ctblkind=0


#%%

# patch_size
# patch_size
# depth_size
def blkextraction(CT,CTsiz1,patch_size,depth_size):   
    CTblocks=[]
    for i in  range(0,CTsiz1[0],patch_size*2):
        for j in range(0,CTsiz1[1],patch_size*2):
            for zi in range(0,CTsiz1[2],depth_size):
                currentBlk=CT[i:i+patch_size*2,j:j+patch_size*2,zi:zi+depth_size]
                currentBlk=currentBlk[:,:,0:depth_size]
                blksiz=currentBlk.shape
                if blksiz[2] == depth_size:
                    CTblocks.append(currentBlk)
    return CTblocks
                
CTblocks=blkextraction(CT,CTsiz1,patch_size,depth_size)
CBblocks=blkextraction(CBCT,CBsiz,patch_size,depth_size)

CTblkLen=len(CTblocks)
CBblkLen=len(CBblocks)
#%%
CBsavepath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/testCB'
CTsavepath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/testCT'
from scipy.io import savemat
os.chdir(CBsavepath)
for filecb in range(CBblkLen):
# filecb=0
    CB_b=CBblocks[filecb] 
    opmatfile="CB-"+str(filecb)+".mat"
    imgdic={"CB_b":CB_b}
    savemat(opmatfile, imgdic)

os.chdir(CTsavepath)
for filect in range(CTblkLen):
# filecb=0
    CT_b=CTblocks[filect] 
    opmatfile="CT-"+str(filect)+".mat"
    imgdic={"CT_b":CT_b}
    savemat(opmatfile, imgdic)