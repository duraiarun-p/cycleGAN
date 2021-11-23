#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:39:25 2021

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

import cv2
def myimgradient(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0) # Find x and y gradients
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    # Find magnitude and angle
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    return magnitude,angle
import scipy.stats
def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    Symmetical Kullback-Leiber Divergence
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance

def JSDlossfx(CB1,CB2,gaussigma):
    
    CB1r=cv2.GaussianBlur(CB1, (7, 7), gaussigma)
    CB2r=cv2.GaussianBlur(CB2, (7, 7), gaussigma)
    Gm1,Ga1=myimgradient(CB1r)
    Gm2,Ga2=myimgradient(CB2r)
    
    
    histogram1, bin_edges1 = np.histogram(Gm1, bins=1000, range=(0, 1))
    histogram2, bin_edges2 = np.histogram(Gm2, bins=1000, range=(0, 1))
    
    JSD=jensen_shannon_distance(histogram1, histogram2)
    return JSD

#%%
mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
cGAN=CycleGAN(mypath,epochs=2,batch_size=5,imgshape=(512,512,1),totalbatchiterations=400)
items=1
SSIM=np.zeros((items,6))

batch_CT, batch_CB =cGAN.dataload()

# CB1=batch_CB[:,:,2]
CB1=batch_CT[:,:,2]
CB2=batch_CB[:,:,3]

gaussigma=2
sigmdel=0.25
eta=0.5
epoch=10000
gaussarr=np.zeros((epoch,1))
JSD11arr=np.zeros((epoch,1))
JSD12arr=np.zeros((epoch,1))
JSDGarr=np.zeros((epoch,1))
J=np.arange(epoch)

for epochi in range(epoch):      
    gaussigmaplusdel=gaussigma+(0.5*sigmdel)
    gaussigmamiusdel=gaussigma-(0.5*sigmdel)
    
    JSD11=JSDlossfx(CB1,CB2,gaussigmaplusdel)
    JSD12=JSDlossfx(CB1,CB2,gaussigmamiusdel)
    JSD11arr[epochi,0]=JSD11
    JSD12arr[epochi,0]=JSD12
    
    Jgrad=(JSD11-JSD12)/sigmdel
    JSDGarr[epochi,0]=Jgrad
    gaussigma=gaussigma+(eta*Jgrad)
    gaussarr[epochi,0]=gaussigma
    



#%%
import matplotlib.pyplot as plt
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("grayscale value")
# plt.ylabel("pixels")
# plt.xlim([0.0, 1.0])  # <- named arguments do not work here
# plt.plot(bin_edges1[0:-1], histogram1,label="hist1")  # <- or here
# plt.plot(bin_edges2[0:-1], histogram2,label="hist2")  # <- or here
# plt.legend()
# plt.show()

plt.figure()
plt.title("Gradient Descent")
plt.xlabel("guassian sigma")
plt.ylabel("JSD")
# plt.xlim([0.0, 1.0])  # <- named arguments do not work here
plt.plot(gaussarr, JSD11arr,label="JSD11")  # <- or here
plt.plot(gaussarr, JSD12arr,label="JSD12")
plt.legend()
plt.show()
#%%
plt.figure()
plt.title("Gradient Descent")
plt.xlabel("epoch")
plt.ylabel("sigma")
# plt.xlim([0.0, 1.0])  # <- named arguments do not work here
plt.plot(J,gaussarr,label="sigma")  # <- or here
# plt.plot(gaussarr, JSD12arr,label="JSD12")
plt.legend()
plt.show()
#%%
plt.figure()
plt.title("Gradient Descent")
plt.xlabel("epoch")
plt.ylabel("JSD")
# plt.xlim([0.0, 1.0])  # <- named arguments do not work here
plt.plot(J, JSD11arr,label="JSD11")  # <- or here
plt.plot(J, JSD12arr,label="JSD12")
plt.legend()
plt.show()
#%%
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)