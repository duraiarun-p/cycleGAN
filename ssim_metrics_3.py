#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:48:42 2021

@author: arun
"""

import time
import datetime
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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
        mat_contents=h5py.File(matfiles[mat_fname_ind])
        # mat_contents_list=list(mat_contents.keys())    
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
        CT=PlanCTCellRef.value
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
            CBCT=CBCellRef.value
            CBCT=np.transpose(CBCT,(2,1,0))
            CBCTs.append(CBCT)
            CBLocRef=mat_contents['CBCTInfocell'][1, CBCTi]
            CBLocRef=mat_contents[CBLocRef]
            CBCTLoc=CBLocRef.value
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

from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

def myimgradient(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0) # Find x and y gradients
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    # Find magnitude and angle
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    return magnitude,angle

def myssim(cb1re,cb2re):  
    cb1re=tf1.expand_dims(CB1r,axis=0)
    cb1re=tf1.expand_dims(cb1re,axis=-1)
    cb2re=tf1.expand_dims(CB2r,axis=0)
    cb2re=tf1.expand_dims(cb2re,axis=-1)
    score1=tf1.image.ssim(cb1re,cb2re,max_val=CB2r.max()-CB2r.min())   
    score2=tf1.image.ssim_multiscale(cb1re,cb2re,max_val=CB2r.max()-CB2r.min()) 
    with tf1.Session() as sess:
        # sess.run(init)
        # sc1=sess.run(score1)
        sc1=score1.eval()
        sc2=score2.eval()
        sc1=np.mean(sc1)
        sc2=np.mean(sc2)
    return sc1, sc2


def mygradssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    Gm1,Ga1=myimgradient(img1)
    Gm2,Ga2=myimgradient(img2)
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    ssim_map_m=ssim_map.mean()
    return ssim_map, ssim_map_m

def myfourcompssim(CB1r,CB2r):
    Gm1,Ga1=myimgradient(CB1r)
    Gm2,Ga2=myimgradient(CB2r)
    score,smap=ssim(CB1r, CB2r, data_range=CB2r.max()-CB2r.min(),full=True)
    maxval=Gm1.max()
    TH1=0.12*maxval
    TH2=0.06*maxval
    imgshape=np.shape(Gm1)
    R1=np.ones((imgshape[0],imgshape[1]))
    R2=np.ones((imgshape[0],imgshape[1]))
    R3=np.zeros((imgshape[0],imgshape[1]))
    R4=np.ones((imgshape[0],imgshape[1])) 
    for i in range(imgshape[0]): 
        for j in range(imgshape[1]):
            if (Gm1[i,j]>TH1) and (Gm2[i,j]>TH1):
                R1[i,j]=0
            # # elif ((Gm1[i,j]>TH1) and (Gm2[i,j]<=TH1) or (Gm1[i,j]<=TH1) and (Gm2[i,j]>TH1)):
            elif (Gm1[i,j]>TH1) and (Gm2[i,j]<=TH1):
                R2[i,j]=0
            # elif (Gm1[i,j]<TH2) and (Gm2[i,j]<=TH1): # this shows some region
            elif (Gm1[i,j]<TH2) and (Gm2[i,j]<TH2):
                R3[i,j]=1
            else:
                R4[i,j]=0
    # scoreR1,R1smap=ssim(smap, R1,data_range=1,full=True)
    # scoreR2,R2smap=ssim(smap, R2,data_range=1,full=True)
    # scoreR3,R3smap=ssim(smap, R3,data_range=1,full=True)
    # scoreR4,R4smap=ssim(smap, R4,data_range=1,full=True)
    R1smap=smap*R1
    R2smap=smap*R2
    R3smap=smap*R3
    R4smap=smap*R4
    Rsmap=0.25*R1smap+0.25*R2smap+0.25*R3smap+0.25*R4smap
    Rscore=np.mean(Rsmap)
    return Rscore
    
def myfourcompgradssim(CB1r,CB2r):
    Gm1,Ga1=myimgradient(CB1r)
    Gm2,Ga2=myimgradient(CB2r)
    score,smap=ssim(Gm1, Gm2,data_range=Gm1.max()-Gm2.min(),full=True)
    maxval=Gm1.max()
    TH1=0.12*maxval
    TH2=0.06*maxval
    imgshape=np.shape(Gm1)
    R1=np.ones((imgshape[0],imgshape[1]))
    R2=np.ones((imgshape[0],imgshape[1]))
    R3=np.zeros((imgshape[0],imgshape[1]))
    R4=np.ones((imgshape[0],imgshape[1])) 
    for i in range(imgshape[0]): 
        for j in range(imgshape[1]):
            if (Gm1[i,j]>TH1) and (Gm2[i,j]>TH1):
                R1[i,j]=0
            # # elif ((Gm1[i,j]>TH1) and (Gm2[i,j]<=TH1) or (Gm1[i,j]<=TH1) and (Gm2[i,j]>TH1)):
            elif (Gm1[i,j]>TH1) and (Gm2[i,j]<=TH1):
                R2[i,j]=0
            # elif (Gm1[i,j]<TH2) and (Gm2[i,j]<=TH1): # this shows some region
            elif (Gm1[i,j]<TH2) and (Gm2[i,j]<TH2):
                R3[i,j]=1
            else:
                R4[i,j]=0
    # scoreR1,R1smap=ssim(smap, R1,data_range=1,full=True)
    # scoreR2,R2smap=ssim(smap, R2,data_range=1,full=True)
    # scoreR3,R3smap=ssim(smap, R3,data_range=1,full=True)
    # scoreR4,R4smap=ssim(smap, R4,data_range=1,full=True)
    R1smap=smap*R1
    R2smap=smap*R2
    R3smap=smap*R3
    R4smap=smap*R4
    Rsmap=0.25*R1smap+0.25*R2smap+0.25*R3smap+0.25*R4smap
    Rscore=np.mean(Rsmap)
    return Rscore

def myfourcompmsssim(CB1r,CB2r):
    Gm1,Ga1=myimgradient(CB1r)
    Gm2,Ga2=myimgradient(CB2r)
    score,smap=ssim(CB1r, CB2r, data_range=CB2r.max()-CB2r.min(),full=True)
    maxval=Gm1.max()
    TH1=0.12*maxval
    TH2=0.06*maxval
    imgshape=np.shape(Gm1)
    R1=np.ones((imgshape[0],imgshape[1]))
    R2=np.ones((imgshape[0],imgshape[1]))
    R3=np.zeros((imgshape[0],imgshape[1]))
    R4=np.ones((imgshape[0],imgshape[1])) 
    for i in range(imgshape[0]): 
        for j in range(imgshape[1]):
            if (Gm1[i,j]>TH1) and (Gm2[i,j]>TH1):
                R1[i,j]=0
            # # elif ((Gm1[i,j]>TH1) and (Gm2[i,j]<=TH1) or (Gm1[i,j]<=TH1) and (Gm2[i,j]>TH1)):
            elif (Gm1[i,j]>TH1) and (Gm2[i,j]<=TH1):
                R2[i,j]=0
            # elif (Gm1[i,j]<TH2) and (Gm2[i,j]<=TH1): # this shows some region
            elif (Gm1[i,j]<TH2) and (Gm2[i,j]<TH2):
                R3[i,j]=1
            else:
                R4[i,j]=0
    # Needs complete re-write
    R1e=tf1.expand_dims(R1,axis=0)
    R1e=tf1.expand_dims(R1e,axis=-1)
    R2e=tf1.expand_dims(R2,axis=0)
    R2e=tf1.expand_dims(R2e,axis=-1)
    R3e=tf1.expand_dims(R3,axis=0)
    R3e=tf1.expand_dims(R3,axis=-1)
    R4e=tf1.expand_dims(R4,axis=0)
    R4e=tf1.expand_dims(R4e,axis=-1)
    smape=tf1.expand_dims(smap,axis=0)
    smape=tf1.expand_dims(smape,axis=-1)
    scoreR1=tf1.image.ssim_multiscale(smape,R1e,max_val=R1.max()-R1.min())
    scoreR2=tf1.image.ssim_multiscale(smape,R2e,max_val=R2.max()-R2.min())
    scoreR3=tf1.image.ssim_multiscale(smape,R3e,max_val=R3.max()-R3.min())
    scoreR4=tf1.image.ssim_multiscale(smape,R4e,max_val=R4.max()-R4.min())
    with tf1.Session() as sess:
        # sess.run(init)
        # sc1=sess.run(score1)
        sc1=scoreR1.eval()
        sc2=scoreR2.eval()
        sc3=scoreR3.eval()
        sc4=scoreR4.eval()
        sc1=sc1.mean()
        sc2=sc2.mean()
        sc3=sc3.mean()
        sc4=sc4.mean()
    Rscore=0.25*sc1+0.25*sc2+0.25*sc3+0.25*sc4
    # Rscore=np.mean(Rsmap)
    return Rscore

def myfourcompgradmsssim(CB1r,CB2r):
    Gm1,Ga1=myimgradient(CB1r)
    Gm2,Ga2=myimgradient(CB2r)
    score,smap=ssim(Gm1, Gm2,data_range=Gm1.max()-Gm2.min(),full=True)
    maxval=Gm1.max()
    TH1=0.12*maxval
    TH2=0.06*maxval
    imgshape=np.shape(Gm1)
    R1=np.ones((imgshape[0],imgshape[1]))
    R2=np.ones((imgshape[0],imgshape[1]))
    R3=np.zeros((imgshape[0],imgshape[1]))
    R4=np.ones((imgshape[0],imgshape[1])) 
    for i in range(imgshape[0]): 
        for j in range(imgshape[1]):
            if (Gm1[i,j]>TH1) and (Gm2[i,j]>TH1):
                R1[i,j]=0
            # # elif ((Gm1[i,j]>TH1) and (Gm2[i,j]<=TH1) or (Gm1[i,j]<=TH1) and (Gm2[i,j]>TH1)):
            elif (Gm1[i,j]>TH1) and (Gm2[i,j]<=TH1):
                R2[i,j]=0
            # elif (Gm1[i,j]<TH2) and (Gm2[i,j]<=TH1): # this shows some region
            elif (Gm1[i,j]<TH2) and (Gm2[i,j]<TH2):
                R3[i,j]=1
            else:
                R4[i,j]=0
    R1e=tf1.expand_dims(R1,axis=0)
    R1e=tf1.expand_dims(R1e,axis=-1)
    R2e=tf1.expand_dims(R2,axis=0)
    R2e=tf1.expand_dims(R2e,axis=-1)
    R3e=tf1.expand_dims(R3,axis=0)
    R3e=tf1.expand_dims(R3,axis=-1)
    R4e=tf1.expand_dims(R4,axis=0)
    R4e=tf1.expand_dims(R4e,axis=-1)
    smape=tf1.expand_dims(smap,axis=0)
    smape=tf1.expand_dims(smape,axis=-1)
    scoreR1=tf1.image.ssim_multiscale(smape,R1e,max_val=R1.max()-R1.min())
    scoreR2=tf1.image.ssim_multiscale(smape,R2e,max_val=R2.max()-R2.min())
    scoreR3=tf1.image.ssim_multiscale(smape,R3e,max_val=R3.max()-R3.min())
    scoreR4=tf1.image.ssim_multiscale(smape,R4e,max_val=R4.max()-R4.min())
    with tf1.Session() as sess:
        # sess.run(init)
        # sc1=sess.run(score1)
        sc1=scoreR1.eval()
        sc2=scoreR2.eval()
        sc3=scoreR3.eval()
        sc4=scoreR4.eval()
        sc1=sc1.mean()
        sc2=sc2.mean()
        sc3=sc3.mean()
        sc4=sc4.mean()
    Rscore=0.25*sc1+0.25*sc2+0.25*sc3+0.25*sc4
    # Rscore=np.mean(Rsmap)
    return Rscore


mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
cGAN=CycleGAN(mypath,epochs=2,batch_size=5,imgshape=(512,512,1),totalbatchiterations=400)
items=1
SSIM=np.zeros((items,6))
for si in range(items):
    
    batch_CT, batch_CB =cGAN.dataload()
    
    # CB1=batch_CB[:,:,2]
    CB1=batch_CT[:,:,2]
    CB2=batch_CB[:,:,3]
    
    # CB1r=cv2.GaussianBlur(CB1, (7, 7), 10)
    # CB2r=cv2.GaussianBlur(CB2, (7, 7), 10)
    
    CB1r=CB1
    CB2r=CB2
    
    SSIMTF,SSIM4TF=myssim(CB1r, CB2r)
    
    SSIMG=mygradssim(CB1r, CB2r)
    
    SSIM4=myfourcompssim(CB1r, CB2r)
    SSIM4G=myfourcompgradssim(CB1r, CB2r)
    SSIM4MS=myfourcompmsssim(CB1r, CB2r)
    SSIM4GMS=myfourcompgradmsssim(CB1r, CB2r)
    
    
    SSIM[si,0]=SSIMTF
    SSIM[si,1]=SSIM4TF
    SSIM[si,2]=SSIM4
    SSIM[si,3]=SSIM4G
    SSIM[si,4]=SSIM4MS
    SSIM[si,5]=SSIM4GMS

#%%
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