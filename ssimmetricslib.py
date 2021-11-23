#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:17:55 2021

@author: arun
"""

import numpy as np
from scipy.ndimage.filters import convolve
import cv2
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage.filters import uniform_filter

#%%
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

    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE) # valid
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(Gm1**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.filter2D(Gm2**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.filter2D(Gm1 * Gm2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2
    
    # sigma1_sq=myvar(Gm1, 11) - mu1_sq
    # sigma2_sq=myvar(Gm2, 11) - mu2_sq
    # sigma12=myvar(np.sqrt(Gm1*Gm2), 11) - mu1_mu2

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
    
    # CBR=0.5*((CB2r.max()-CB2r.min())+(CB1r.max()-CB1r.min()))
    
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
    
    sigma1_sq = cv2.filter2D(Gm1**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.filter2D(Gm2**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.filter2D(Gm1 * Gm2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2

    # sigma1_sq=myvar(Gm1, 11)
    # sigma2_sq=myvar(Gm2, 11)
    # sigma12=myvar(np.sqrt(Gm1*Gm2), 11)
    
    # sigma1_sq=myvar(Gm1, 11) - mu1_sq
    # sigma2_sq=myvar(Gm2, 11) - mu2_sq
    # sigma12=myvar(np.sqrt(Gm1*Gm2), 11) - mu1_mu2
    
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