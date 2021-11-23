#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:31:35 2021

@author: arun
"""
print('The code has started')

import time
import datetime
import h5py
import numpy as np
# from scipy import signal
# from scipy.ndimage.filters import convolve
from os import listdir
from os.path import isfile, join
import cv2

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

# import os
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["H5PY_DEFAULT_READONLY"] = 1

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
#%%
class CycleGAN():
    
    def dataload(self):
        # mypath=self.DataPath
        onlyfiles = [f for f in listdir(self.DataPath) if isfile(join(self.DataPath, f))]
        onlyfiles.sort()
        onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
        onlyfiles = onlyfiles[0:-onlyfileslenrem]
        matfiles=[join(self.DataPath,f) for f in onlyfiles]
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
        PlanCTLoc=PlanCTLocRef[()]
        PlanCTCellRef=mat_contents['CTInfoCell'][2, CTindex]
        PlanCTCellRef=mat_contents[PlanCTCellRef]
        CT=PlanCTCellRef[()]
        CT=np.transpose(CT,(2,1,0))
        CTsiz1=CT.shape
        CT_rand_index=np.random.choice(CTsiz1[2],size=self.batch_size,replace=False)
        batch_CT_img=np.zeros((CTsiz1[0],CTsiz1[1],len(CT_rand_index)))
        for ri in range(len(CT_rand_index)):
            batch_CT_img[:,:,ri]=CT[:,:,CT_rand_index[ri]]    
        CBCTCellRef=mat_contents['CBCTInfocell']
        CBCLen=np.shape(CBCTCellRef)
        CBCTs=[]
        for CBCTi in range(CBCLen[1]):
            # print(CBCTi)
            CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
            CBCellRef=mat_contents[CBCellRef]
            CBCT=CBCellRef[()]
            CBCT=np.transpose(CBCT,(2,1,0))
            CBCTs.append(CBCT)
            CBLocRef=mat_contents['CBCTInfocell'][1, CBCTi]
            CBLocRef=mat_contents[CBLocRef]
            CBCTLoc=CBLocRef[()]
        CBsiz=CBCT.shape
        batch_CB_img=np.zeros((CTsiz1[0],CTsiz1[1],self.batch_size))
        for cbi in range(self.batch_size):
            CB_rand_sl_index=np.random.choice(CBsiz[2])
            CB_rand_pat_index=np.random.choice(CBCLen[1],replace=False)
            # print(CB_rand_pat_index)
            # print(CB_rand_sl_index)
            batch_CB_img[:,:,cbi]=CBCTs[CB_rand_pat_index][:,:,CB_rand_sl_index]
        del mat_contents
        return batch_CT_img, batch_CB_img
    
    def __init__(self,mypath,epochs,batch_size,imgshape,totalbatchiterations):
        self.DataPath=mypath
        self.batch_size=batch_size
        self.img_shape=imgshape
        self.input_shape=tuple([batch_size,imgshape])
        self.genafilter = 32
        self.discfilter = 64
        self.epochs = epochs
        self.totalbatchiterations=totalbatchiterations
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.9 * self.lambda_cycle    # Identity loss

#%%        

import ssimmetricslib as ssm
# from ssimmetrics import myfourcompssim

mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
cGAN=CycleGAN(mypath,epochs=2,batch_size=5,imgshape=(512,512,1),totalbatchiterations=400)

cycles=1

scores=np.zeros((cycles,8))

for scorei in range(cycles):
    

    batch_CT, batch_CB =cGAN.dataload()
    
    # CB1=batch_CB[:,:,2]
    CB1=batch_CT[:,:,2]
    CB2=batch_CB[:,:,3]
    
    gausssigma=0.5
    CB1r=cv2.GaussianBlur(CB1, (7, 7), gausssigma)
    CB2r=cv2.GaussianBlur(CB2, (7, 7), gausssigma)
    

    # CB1r=CB2
    
    # CB2r=CB1r
    # CB1r=CB1
    
    
    # Gm1,Ga1=ssm.myimgradient(CB1r)
    # Gm2,Ga2=myimgradient(CB2r)
    
    # CBR=0.5*((CB2r.max()-CB2r.min())+(CB1r.max()-CB1r.min()))
    
    # GMR=0.5*((Gm1.max()-Gm1.min())+(Gm2.max()-Gm2.min()))
    # score,smap=ssim(CB1r, CB2r, data_range=CBR,full=True)
    # score0,smap0=ssim(Gm1, Gm2, data_range=GMR,full=True)
    smap1,score1=ssm.mynewssim(CB1r, CB2r)
    smap2,score2=ssm.mynewssimgrad(CB1r, CB2r) #Gradient is much better
    smap3,score3=ssm.myfourcompssim(CB1r, CB2r)
    smap4,score4=ssm.myfourcompgradssim(CB1r, CB2r)
    score5=ssm.mymultiscalessim(CB1r, CB2r)
    score6=ssm.mymultiscalegradssim(CB1r, CB2r)
    score7=ssm.myfourcompmultiscalessim(CB1r, CB2r)
    score8=ssm.myfourcompmultiscalegradssim(CB1r, CB2r)
    
    scores[scorei:]=[score1,score2,score3,score4,score5,score6,score7,score8]
    sc=np.mean(scores, axis=0)
#%%
# print("SSIM=%s, G-SSIM=%s 4-SSIM=%s, 4-G-SSIM=%s MS=%s, MS-G=%s, 4-MS=%s, 4-MS-G=%s"%(score1,score2,score5,score6,score3,score4,score7,score8))
# batch_CT, batch_CB =cGAN.dataload()
#%%
# batch_CT = np.transpose(batch_CT,(2,0,1))
# batch_CT = np.expand_dims(batch_CT, -1)
# batch_CB = np.transpose(batch_CB,(2,0,1))
# batch_CB = np.expand_dims(batch_CB, -1)
# batch_CB = np.squeeze(batch_CB, axis=3)
# batch_CB = np.transpose(batch_CB,(1,2,0))
# batch_CT = np.squeeze(batch_CT, axis=3)
# batch_CT = np.transpose(batch_CT,(1,2,0))
#%%
import matplotlib.pyplot as plt
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(CB1r, cmap='gray'),plt.show()
plt.subplot(1,2,2)
plt.imshow(CB2r, cmap='gray'),plt.show()
#%%
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)