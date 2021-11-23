#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 12:34:05 2021

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
mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
cGAN=CycleGAN(mypath,epochs=2,batch_size=5,imgshape=(512,512,1),totalbatchiterations=400)

batch_CT, batch_CB =cGAN.dataload()

CB1=batch_CB[:,:,2]
# CB1=batch_CT[:,:,2]
CB2=batch_CB[:,:,3]
# CB1=CB2
Cb1=np.linalg.norm(CB1)

# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim
# from skimage import exposure
# image = exposure.rescale_intensity(CB1, in_range=(0, 1))

import cv2


import matplotlib.pyplot as plt
#%%
CB1r=cv2.GaussianBlur(CB1, (7, 7), 10)
CB2r=cv2.GaussianBlur(CB2, (7, 7), 10)
# CB1r=CB2
# score,smap=compare_ssim(CB1r, CB2r, full=True)


#%%
# from sewar.full_ref import msssim

# CB2re=CB2r.astype('float32')
# CB1re=CB1r.astype('float32')

# # score11=msssim(CB1re,CB2re)


from skvideo.measure import msssim
CB2re=np.expand_dims(CB2r, axis=0)
CB2re=np.expand_dims(CB2re, axis=-1)
CB1re=np.expand_dims(CB1r, axis=0)
CB1re=np.expand_dims(CB1re, axis=-1)

score1, smap1 = msssim(CB1re,CB2re)


import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

    
def myimgradient(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0) # Find x and y gradients
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    # Find magnitude and angle
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    return magnitude,angle

Gm1,Ga1=myimgradient(CB1r)
Gm2,Ga2=myimgradient(CB2r)

score,smap=ssim(CB1r, CB2r,data_range=CB2r.max()-CB2r.min(),full=True)
# score,smap=ssim(Gm1, Gm2,data_range=Gm1.max()-Gm2.min(),full=True)
imgplt=plt.imshow(smap, cmap='gray'),plt.show()

#%%
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


maxval=Gm1.max()

# TH1=0.12*maxval
# TH2=0.06*maxval

TH1=0.5*maxval
TH2=0.06*maxval

th,Gm1t=cv2.threshold(Gm1, TH1, maxval, cv2.THRESH_BINARY)

# sb1,sb2=myssim(CB1, CB2)

# sc1,sc2=myssim(CB1r, CB2r)

# sgm1,sgm2=myssim(Gm1, Gm2)
# sga1,sga2=myssim(Ga1, Ga2)

# sg1=np.mean([sgm1,sga1])
# sg2=np.mean([sgm2,sga2])

imgshape=np.shape(CB1)

R1=np.ones((imgshape[0],imgshape[1]))
R2=np.ones((imgshape[0],imgshape[1]))
R3=np.zeros((imgshape[0],imgshape[1]))
R4=np.ones((imgshape[0],imgshape[1]))
# R1=np.zeros((imgshape[0],imgshape[1]))
# R2=np.zeros((imgshape[0],imgshape[1]))
# R3=np.zeros((imgshape[0],imgshape[1]))
# R4=np.zeros((imgshape[0],imgshape[1]))

for i in range(imgshape[0]): 
    for j in range(imgshape[1]):
        if (Gm1[i,j]>TH1) and (Gm2[i,j]>TH1):
            R1[i,j]=0
        # # elif ((Gm1[i,j]>TH1) and (Gm2[i,j]<=TH1) or 
        elif (Gm1[i,j]<=TH1) and (Gm2[i,j]>TH1):
        # elif (Gm1[i,j]>TH1) and (Gm2[i,j]<=TH1):
            R2[i,j]=0
        # elif (Gm1[i,j]<TH2) and (Gm2[i,j]<=TH1): # this shows some region
        elif (Gm1[i,j]<TH2) and (Gm2[i,j]<TH2):
            R3[i,j]=1
        else:
            R4[i,j]=0
            
# R2=R1   
 ##%%
import matplotlib.pyplot as plt
# import matplotlib.figure as fig
# fig, axs = plt.subplots(2)
# imgplt=plt.imshow(CB2r, cmap='gray'),plt.show()
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(Gm1, cmap='gray'),plt.show()
plt.subplot(1,2,2)
plt.imshow(Gm2, cmap='gray'),plt.show()

plt.figure(3)
plt.subplot(2,2,1)
plt.imshow(R1, cmap='gray'),plt.show()
plt.subplot(2,2,2)
plt.imshow(R2, cmap='gray'),plt.show()
plt.subplot(2,2,3)
plt.imshow(R3, cmap='gray'),plt.show()
plt.subplot(2,2,4)
plt.imshow(R4, cmap='gray'),plt.show()


R1smap=smap*R1
R2smap=smap*R2
R3smap=smap*R3
R4smap=smap*R4

# scoreR1,R1smap=ssim(smap, R1,data_range=1,full=True)
# scoreR2,R2smap=ssim(smap, R2,data_range=1,full=True)
# scoreR3,R3smap=ssim(smap, R3,data_range=1,full=True)
# scoreR4,R4smap=ssim(smap, R4,data_range=1,full=True)

# score2=tf1.image.ssim_multiscale(smap,R1,max_val=smap.max()-smap.min())

plt.figure(4)
plt.subplot(2,2,1)
plt.imshow(R1smap, cmap='gray'),plt.show()
plt.subplot(2,2,2)
plt.imshow(R2smap, cmap='gray'),plt.show()
plt.subplot(2,2,3)
plt.imshow(R3smap, cmap='gray'),plt.show()
plt.subplot(2,2,4)
plt.imshow(R4smap, cmap='gray'),plt.show()

Rsmap=0.25*R1smap+0.25*R2smap+0.25*R3smap+0.25*R4smap
# Rsmap=0.25*(R1smap+R2smap)+0.25*R3smap+0.25*R4smap
# Rsmap=R1smap+R2smap+R3smap+R4smap

# Rscore=0.25*R1smap.mean()+0.25*R2smap.mean()+0.25*R3smap.mean()+0.25*R4smap.mean()
# Rscore=Rsmap.mean()*1.3333
Rscore=Rsmap.mean()

plt.figure(5)
# plt.subplot(2,2,1)
plt.imshow(Rsmap, cmap='gray'),plt.show()

#%%  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)