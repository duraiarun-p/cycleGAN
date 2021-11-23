#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 21:46:08 2021

@author: arun
"""


import time
import datetime
import h5py
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
from os import listdir
from os.path import isfile, join
import cv2

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


from scipy.ndimage.filters import uniform_filter

def myvar(X, window_size):
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return (c2 - c1*c1)


def mynewssim(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    C1m=(np.max(img1)-np.min(img1))
    C2m=(np.max(img2)-np.min(img2))
    Cmr=0.5*(C1m+C2m)
    # C1m=0.5(img1.max()+img2.max())
    # C1 = (0.01 * 255)**2
    # C2 = (0.03 * 255)**2
    C1 = (0.01 * Cmr)**2
    C2 = (0.03 * Cmr)**2
    
    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE)  # valid
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2
    
    # mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE)[5:-5, 5:-5]  # valid
    # mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)[5:-5, 5:-5]
    # mu1_sq = mu1**2
    # mu2_sq = mu2**2
    # mu1_mu2 = mu1 * mu2
    # sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=cv2.BORDER_REPLICATE)[5:-5, 5:-5] - mu1_sq
    # sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=cv2.BORDER_REPLICATE)[5:-5, 5:-5] - mu2_sq
    # sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REPLICATE)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_index=ssim_map.mean()
    return ssim_map,ssim_index

def myimgradient(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0) # Find x and y gradients
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    # Find magnitude and angle
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    return magnitude,angle

def mynewssimgrad(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    # window1=np.zeros((11,11))
    C1m=(np.max(img1)-np.min(img1))
    C2m=(np.max(img2)-np.min(img2))
    Cmr=0.5*(C1m+C2m)
    
    Gm1,Ga1=myimgradient(img1)
    Gm2,Ga2=myimgradient(img2)
    G1m=(np.max(Gm1)-np.min(Gm1))
    G2m=(np.max(Gm2)-np.min(Gm2))
    Gmr=0.5*(G1m+G2m)
    
    C1 = (0.01 * Cmr)**2
    C2 = (0.03 * Cmr)**2
    
    # mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE)[5:-5, 5:-5]  # valid
    # mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)[5:-5, 5:-5]
    # mu1_sq = mu1**2
    # mu2_sq = mu2**2
    # mu1_mu2 = mu1 * mu2
    
    # sigma1_sq=myvar(Gm1, 11)[5:-5, 5:-5]
    # sigma2_sq=myvar(Gm2, 11)[5:-5, 5:-5]
    # sigma12=myvar(np.sqrt(Gm1*Gm2), 11)[5:-5, 5:-5]

    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE) # valid
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq=myvar(Gm1, 11)
    sigma2_sq=myvar(Gm2, 11)
    sigma12=myvar(np.sqrt(Gm1*Gm2), 11)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_index=ssim_map.mean()
    return ssim_map,ssim_index

def myfourcompssim(CB1r,CB2r):
    Gm1,Ga1=myimgradient(CB1r)
    Gm2,Ga2=myimgradient(CB2r)
    
    CBR=0.5*((CB2r.max()-CB2r.min())+(CB1r.max()-CB1r.min()))
    
    score,smap=ssim(CB1r, CB2r, data_range=CBR,full=True)
    
    # smap,score=mynewssim(CB1r, CB2r)
    
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

    R1smap=smap*R1
    R2smap=smap*R2
    R3smap=smap*R3
    R4smap=smap*R4
    Rsmap=0.25*R1smap+0.25*R2smap+0.25*R3smap+0.25*R4smap
    Rscore=np.mean(Rsmap)
    return Rsmap, Rscore

def myfourcompgradssim(CB1r,CB2r):
    Gm1,Ga1=myimgradient(CB1r)
    Gm2,Ga2=myimgradient(CB2r)
    
    CBR=0.5*((CB2r.max()-CB2r.min())+(CB1r.max()-CB1r.min()))
    
    # score,smap=ssim(CB1r, CB2r, data_range=CBR,full=True)
    
    smap,score=mynewssimgrad(CB1r, CB2r)
    
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

    R1smap=smap*R1
    R2smap=smap*R2
    R3smap=smap*R3
    R4smap=smap*R4
    Rsmap=0.25*R1smap+0.25*R2smap+0.25*R3smap+0.25*R4smap
    Rscore=np.mean(Rsmap)
    return Rsmap, Rscore


def ssimformulti(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    C1m=(np.max(img1)-np.min(img1))
    C2m=(np.max(img2)-np.min(img2))
    Cmr=0.5*(C1m+C2m)
    
    C1 = (0.01 * Cmr)**2
    C2 = (0.03 * Cmr)**2
    
    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE)  # valid
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    cs_map = (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    
    ssim_index=ssim_map.mean()
    return ssim_map,cs_map,ssim_index

# from scipy.ndimage.filters import convolve

def mymultiscalessim(im1, im2):
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    # im1 = img1.astype(np.float64)
    # im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map, ssim_index = ssimformulti(im1, im2)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = convolve(im1, downsample_filter, mode='reflect')
        filtered_im2 = convolve(im2, downsample_filter, mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    
    mulssim=(np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))
    return mulssim

def gradssimformulti(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    Gm1,Ga1=myimgradient(img1)
    Gm2,Ga2=myimgradient(img2)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    C1m=(np.max(img1)-np.min(img1))
    C2m=(np.max(img2)-np.min(img2))
    Cmr=0.5*(C1m+C2m)
    
    C1 = (0.01 * Cmr)**2
    C2 = (0.03 * Cmr)**2
    
    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE)  # valid
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq=myvar(Gm1, 11)
    sigma2_sq=myvar(Gm2, 11)
    sigma12=myvar(np.sqrt(Gm1*Gm2), 11)
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    cs_map = (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    
    ssim_index=ssim_map.mean()
    return ssim_map,cs_map,ssim_index


def mymultiscalegradssim(im1, im2):
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    # im1 = img1.astype(np.float64)
    # im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map, ssim_index = gradssimformulti(im1, im2)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = convolve(im1, downsample_filter, mode='reflect')
        filtered_im2 = convolve(im2, downsample_filter, mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    
    mulssim=(np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))
    return mulssim

def myfourcompmultiscalessim(im1,im2):
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    # im1 = img1.astype(np.float64)
    # im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map, ssim_index = ssimformulti(im1, im2)
        Gm1,Ga1=myimgradient(im1)
        Gm2,Ga2=myimgradient(im2)
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
    
        R1smap=ssim_map*R1
        R2smap=ssim_map*R2
        R3smap=ssim_map*R3
        R4smap=ssim_map*R4
        ssim_map=0.25*R1smap+0.25*R2smap+0.25*R3smap+0.25*R4smap
        # Rscore=np.mean(Rsmap)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = convolve(im1, downsample_filter, mode='reflect')
        filtered_im2 = convolve(im2, downsample_filter, mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    
    mulssim=(np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))
    
    return mulssim

def myfourcompmultiscalegradssim(im1,im2):
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    # im1 = img1.astype(np.float64)
    # im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map, ssim_index = gradssimformulti(im1, im2)
        Gm1,Ga1=myimgradient(im1)
        Gm2,Ga2=myimgradient(im2)
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
    
        R1smap=ssim_map*R1
        R2smap=ssim_map*R2
        R3smap=ssim_map*R3
        R4smap=ssim_map*R4
        ssim_map=0.25*R1smap+0.25*R2smap+0.25*R3smap+0.25*R4smap
        # Rscore=np.mean(Rsmap)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = convolve(im1, downsample_filter, mode='reflect')
        filtered_im2 = convolve(im2, downsample_filter, mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    
    mulssim=(np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))
    
    return mulssim

#%%
from skimage.metrics import structural_similarity as ssim

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
    
    
    # Gm1,Ga1=myimgradient(CB1r)
    # Gm2,Ga2=myimgradient(CB2r)
    
    CBR=0.5*((CB2r.max()-CB2r.min())+(CB1r.max()-CB1r.min()))
    
    # GMR=0.5*((Gm1.max()-Gm1.min())+(Gm2.max()-Gm2.min()))
    # score,smap=ssim(CB1r, CB2r, data_range=CBR,full=True)
    # score0,smap0=ssim(Gm1, Gm2, data_range=GMR,full=True)
    smap1,score1=mynewssim(CB1r, CB2r)
    smap2,score2=mynewssimgrad(CB1r, CB2r) #Gradient is much better
    smap3,score3=myfourcompssim(CB1r, CB2r)
    smap4,score4=myfourcompgradssim(CB1r, CB2r)
    score5=mymultiscalessim(CB1r, CB2r)
    score6=mymultiscalegradssim(CB1r, CB2r)
    score7=myfourcompmultiscalessim(CB1r, CB2r)
    score8=myfourcompmultiscalegradssim(CB1r, CB2r)
    
    scores[scorei:]=[score1,score2,score3,score4,score5,score6,score7,score8]
    sc=np.mean(scores, axis=0)
#%%
# print("SSIM=%s, G-SSIM=%s 4-SSIM=%s, 4-G-SSIM=%s MS=%s, MS-G=%s, 4-MS=%s, 4-MS-G=%s"%(score1,score2,score5,score6,score3,score4,score7,score8))

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