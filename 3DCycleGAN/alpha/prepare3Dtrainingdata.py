#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:49:40 2021

@author: arun
"""

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
from scipy.io import savemat

#%%

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

def datasave(onlyfiles,CBtrainpath,CTtrainpath):
    if isinstance(onlyfiles,list)==True:    
        matfiles=[join(mypath,f) for f in onlyfiles]
    else:
        matfiles=[join(mypath,onlyfiles)]
    # mat_fname_ind=np.random.choice(len(matfiles),replace=False)
    for mat_fname_ind in range(len(matfiles)):
        mat_contents=h5py.File(matfiles[mat_fname_ind],'r')
        # mat_contents=h5py.File(matfile,'r')
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
        CT = (CT-np.min(CT))/(np.max(CT)-np.min(CT))
        CTsiz1=CT.shape
        # CTblocksN=(CTsiz1[0]*CTsiz1[1]*CTsiz1[2])//(patch_size*patch_size*4*depth_size)
        CTblocks=blkextraction(CT,CTsiz1,patch_size,depth_size)
        CTblkLen=len(CTblocks)
        os.chdir(CTtrainpath)
        for filect in range(CTblkLen):
        # filecb=0
            CT_b=CTblocks[filect] 
            opmatfile="CT-"+str(mat_fname_ind)+"-"+str(filect)+".mat"
            imgdic={"CT_b":CT_b}
            savemat(opmatfile, imgdic)
        
        CBCTCellRef=mat_contents['CBCTInfocell']
        CBCLen=np.shape(CBCTCellRef)
        # CBCTs=[]
        # CBCTi=np.random.choice(CBCLen[1],replace=False)
        for CBCTi in range(CBCLen[1]):  
            CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
            CBCellRef=mat_contents[CBCellRef]
            CBCT=CBCellRef[()]
            CBCT=np.transpose(CBCT,(2,1,0))
            CBCT = (CBCT-np.min(CBCT))/(np.max(CBCT)-np.min(CBCT))
            CBsiz=CBCT.shape
            # CBblocksN=(CBsiz[0]*CBsiz[1]*CBsiz[2])//(patch_size*patch_size*4*depth_size)
            CBblocks=blkextraction(CBCT,CBsiz,patch_size,depth_size)
            CBblkLen=len(CBblocks)
            os.chdir(CBtrainpath)
            for filecb in range(CBblkLen):
            # filecb=0
                CB_b=CBblocks[filecb] 
                opmatfile="CB-"+str(mat_fname_ind)+"-S-"+str(CBCTi)+"-"+str(filecb)+".mat"
                imgdic={"CB_b":CB_b}
                savemat(opmatfile, imgdic)
        print('Data saved')
#%%
mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
depth_size=32
patch_size=16

CBtrainpath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/trainCB'
CTtrainpath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/trainCT'


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()
onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
onlyfiles = onlyfiles[:-onlyfileslenrem]
datasave(onlyfiles,CBtrainpath,CTtrainpath)
print('Training Data saved')
CBtrainpath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/validCB'
CTtrainpath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/validCT'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()
onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
onlyfiles = onlyfiles[-onlyfileslenrem]
datasave(onlyfiles,CBtrainpath,CTtrainpath)
print('Validation data saved')
CBtrainpath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/testCB'
CTtrainpath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/testCT'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()
onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
onlyfiles = onlyfiles[-1:]
datasave(onlyfiles,CBtrainpath,CTtrainpath)
print('Testing data saved')