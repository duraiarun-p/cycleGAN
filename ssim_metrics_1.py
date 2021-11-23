#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 17:20:48 2021

@author: arun
"""

import cv2
import numpy as np


import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
#%%

CB1r=cv2.imread('s1.jpg')
CB2r=cv2.imread('s2.jpg')

CB1r=CB1r[:,:,2]
CB2r=CB2r[:,:,2]

cbs1=CB1r.shape
cbs=np.zeros((2,1))
cbs[0]=cbs1[1]
cbs[1]=cbs1[0]
cbs=np.transpose(cbs)
# # cbs=tuple(cbs)

CB2r=cv2.resize(CB2r, cbs1, interpolation= cv2.INTER_NEAREST)
CB2r=cv2.transpose(CB2r)

#%%
def myimgradient(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0) # Find x and y gradients
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    # Find magnitude and angle
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    return magnitude,angle

Gm1,Ga1=myimgradient(CB1r)
Gm2,Ga2=myimgradient(CB2r)
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

sc1,sc2=myssim(CB1r, CB2r)

sgm1,sgm2=myssim(Gm1, Gm2)
sga1,sga2=myssim(Ga1, Ga2)

sg1=np.mean([sgm1,sga1])
sg2=np.mean([sgm2,sga2])


#%%
import matplotlib.pyplot as plt
# import matplotlib.figure as fig
# fig, axs = plt.subplots(2)
# imgplt=plt.imshow(CB2r, cmap='gray'),plt.show()
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(CB1r, cmap='gray'),plt.show()
plt.subplot(1,2,2)
plt.imshow(CB2r, cmap='gray'),plt.show()

plt.figure(3)
plt.subplot(1,2,1)
plt.imshow(Gm1, cmap='gray'),plt.show()
plt.subplot(1,2,2)
plt.imshow(Ga1, cmap='gray'),plt.show()
