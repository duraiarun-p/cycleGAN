#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 18:54:15 2021

@author: arun
"""

import time
import datetime
import cv2
import itk
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import os
import sys
from scipy.io import savemat

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import ssimetricTFlib as ssTF

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()

scriptpath=os.getcwd()

sys.path.append(scriptpath)
#%%

class CycleGAN():     

    
    def printmypath(self):
        print(self.DataPath)
        
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
        # PlanCTLoc=PlanCTLocRef[()]
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
            # CBCTLoc=CBLocRef[()]
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
    
    @staticmethod
    def conv2d(layer_input, filters, f_size=4,stride=2,normalization=True):
        """Discriminator layer"""
        d = layers.Conv2D(filters, kernel_size=f_size,strides=stride, padding='same',activation='relu')(layer_input)
        if normalization:
            d = InstanceNormalization()(d)
        d = layers.LeakyReLU(alpha=0.2)(d)
        return d
    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, stride=1, dropout_rate=0,skip=True):
          """Layers used during upsampling"""
          u = layers.UpSampling2D(size=2)(layer_input)
          u = layers.Conv2D(filters, kernel_size=f_size, strides=stride, padding='same', activation='tanh')(u)
          if dropout_rate:
              u = layers.Dropout(dropout_rate)(u)
          u = InstanceNormalization()(u)
          if skip:
              u = layers.Concatenate()([u, skip_input])
          return u
    
    def build_discriminator(self):
        d0 = keras.Input(shape=self.img_shape,name='Input')
        d1 = self.conv2d(d0, self.discfilter, stride=2, normalization=False)
        d2 = self.conv2d(d1, self.discfilter*2, stride=2, normalization=True)
        d3 = self.conv2d(d2, self.discfilter*4, stride=2, normalization=True)
        d4 = self.conv2d(d3, self.discfilter*8, stride=2, normalization=True)
        d5 = self.conv2d(d4, self.discfilter*8, stride=1, normalization=True)
        d6 = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)
        
        return keras.Model(d0,d6)
    
    def build_generator(self):
        d0 = keras.Input(shape=self.img_shape,name='Input')
        d1 = self.conv2d(d0, self.genafilter,stride=1,normalization=True)
        d2 = self.conv2d(d1, self.genafilter,stride=2,normalization=True)
        d3 = self.conv2d(d2, self.genafilter*2,stride=1,normalization=True)
        d4 = self.conv2d(d3, self.genafilter*2,stride=2,normalization=True)
        d5 = self.conv2d(d4, self.genafilter*4,stride=1,normalization=True)
        d6 = self.conv2d(d5, self.genafilter*4,stride=2,normalization=True)
        d7 = self.conv2d(d6, self.genafilter*8,stride=1,normalization=True)
        d8 = self.conv2d(d7, self.genafilter*8,stride=2,normalization=True)
        d9 = self.conv2d(d8, self.genafilter*16,stride=1,normalization=True)
        d10 = self.conv2d(d9, self.genafilter*16,stride=2,normalization=True)
        
        u10 = self.deconv2d(d10, d8, self.genafilter*8,stride=1)
        u9 = self.conv2d(u10, self.genafilter*8,stride=1,normalization=True)
        u8 = self.deconv2d(u9, d6, self.genafilter*4,stride=1)
        u7 = self.conv2d(u8, self.genafilter*4,stride=1,normalization=True)
        u6 = self.deconv2d(u7, d4, self.genafilter*2,stride=1)
        u5 = self.conv2d(u6, self.genafilter*2,stride=1,normalization=True)
        u4 = self.deconv2d(u5, d2, self.genafilter,stride=1)
        u3 = self.conv2d(u4, self.genafilter,stride=1,normalization=True)
        u2 = self.deconv2d(u3,d1, self.genafilter,stride=1,skip=False)
        u1 = layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(u2)
        
        return keras.Model(d0,u1)
    
    def predictCT2CB(self,batch_CT,batch_CB,GenCT2CBWeightsFileName):
        batch_CT = np.transpose(batch_CT,(2,0,1))
        batch_CT = np.expand_dims(batch_CT, -1)
        batch_CB = np.transpose(batch_CB,(2,0,1))
        batch_CB = np.expand_dims(batch_CB, -1)
        
        # GenCT2CBWeightsFileName="GenCT2CBWeights.h5"
        TestGenCT2CB=self.build_generator()
        TestGenCT2CB.load_weights(GenCT2CBWeightsFileName)
        batch_CB_P=TestGenCT2CB.predict(batch_CT)
        batch_CB_P=np.squeeze(batch_CB_P,axis=3)
        batch_CB_P=np.transpose(batch_CB_P,(1,2,0))
        
        return batch_CB_P
    
    def __init__(self,mypath,weightoutputpath,epochs,batch_size,imgshape,totalbatchiterations):
         self.DataPath=mypath
         self.WeightsPath=weightoutputpath
         self.batch_size=batch_size
         self.img_shape=imgshape
         self.input_shape=tuple([batch_size,imgshape])
         self.genafilter = 32
         self.discfilter = 64
         self.epochs = epochs
         self.totalbatchiterations=totalbatchiterations
         self.lambda_cycle = 10.0                    # Cycle-consistency loss
         self.lambda_id = 0.9 * self.lambda_cycle    # Identity loss
         
         os.chdir(self.WeightsPath)
         
         # optimizer = keras.optimizers.Adam(0.0002, 0.5)

        
#%%

mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
outputpath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/output'
weightoutputpath='/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/1892021/'
# imgshape=(512,512)

# batch_size=1
# epochs=1
cGAN=CycleGAN(mypath,weightoutputpath,epochs=2,batch_size=1,imgshape=(512,512,1),totalbatchiterations=40)

# batch_CT, batch_CB, G_losses, D_losses=cGAN.traincgan()

batch_CT, batch_CB =cGAN.dataload()
gausssigma=0.5
batch_CTs=cv2.GaussianBlur(batch_CT, (7, 7), gausssigma)
batch_CBs=cv2.GaussianBlur(batch_CB, (7, 7), gausssigma)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(batch_CT, cmap='gray'),plt.show()
plt.subplot(1,2,2)
plt.imshow(batch_CB, cmap='gray'),plt.show()

batch_CT = np.transpose(batch_CT,(2,0,1))
batch_CT = np.expand_dims(batch_CT, -1)
batch_CB = np.transpose(batch_CB,(2,0,1))
batch_CB = np.expand_dims(batch_CB, -1)




plt.figure(5)
plt.subplot(1,2,1)
plt.imshow(batch_CTs, cmap='gray'),plt.show()
plt.subplot(1,2,2)
plt.imshow(batch_CBs, cmap='gray'),plt.show()



# batch_CT=batch_CT1[:,:,2]
# batch_CB=batch_CB1[:,:,2]

batch_CTs = np.expand_dims(batch_CTs, -1)
batch_CTs = np.expand_dims(batch_CTs, 0)
batch_CBs = np.expand_dims(batch_CBs, -1)
batch_CBs = np.expand_dims(batch_CBs, 0)


imgshape=(512,512,1)
#%%

import cycleganssimetriclib as tfs

# batch_CB=batch_CT
filter_sigma=0.5
filter_size=11
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
# power_factors =_MSSSIM_WEIGHTS

max_val=0.5*((batch_CT.max()-batch_CT.min())+(batch_CB.max()-batch_CB.min()))

score0=tfs.tfssim(batch_CT, batch_CB, max_val,filter_size,filter_sigma)
print('TF inbuilt:')
print(score0)

score1,smap1=tfs.tfssim_custom(batch_CT, batch_CB, max_val,filter_size,filter_sigma)
score1=score1-1
print('TF custom:')
print(score1)

score2,smap2=tfs.tfssim4c(batch_CT, batch_CB, max_val,filter_size,filter_sigma)
print('TF 4-c custom:')
print(score2)

score3,smap3=tfs.tfssim4cg(batch_CT, batch_CB, max_val,filter_size,filter_sigma)
print('TF 4-c-G custom:')
print(score3)



#%%
score4=tfs.tfmsssim(batch_CT, batch_CB, max_val,filter_size,filter_sigma)
print('TF MS inbuilt:')
print(score4)

score5=tfs.tfmssim_custom(batch_CT, batch_CB, max_val,_MSSSIM_WEIGHTS,filter_size,filter_sigma)
print('TF MS custom:')
print(score5)

score6=tfs.tfmssim_4c(batch_CT, batch_CB, max_val,_MSSSIM_WEIGHTS,filter_size,filter_sigma)
print('TF MS 4-C custom:')
print(score6)

score7=tfs.tfmssim_4cg(batch_CT, batch_CB, max_val,_MSSSIM_WEIGHTS,filter_size,filter_sigma)
print('TF MS 4-C-G custom:')
print(score7)

#%%


def viewsmap(smap,fignum):
    smap=np.squeeze(smap,axis=3)
    smap=np.transpose(smap,(1,2,0))
    plt.figure(fignum),plt.imshow(smap, cmap='gray'),plt.show()
    
viewsmap(smap1,2)
viewsmap(smap2,3)
viewsmap(smap3,4)

#%%

#%%
# max_vals=0.5*((batch_CTs.max()-batch_CTs.min())+(batch_CBs.max()-batch_CBs.min()))


# score0=tfs.tfssim(batch_CTs, batch_CBs, max_vals,filter_size,filter_sigma)
# print('TF inbuilt:')
# print(score0)

# score1,smap1=tfs.tfssim_custom(batch_CTs, batch_CBs, max_vals,filter_size,filter_sigma)
# print('TF custom:')
# print(score1)

# score2,smap2=tfs.tfssim4c(batch_CTs, batch_CBs, max_vals,filter_size,filter_sigma)
# print('TF 4-c custom:')
# print(score2)

# score3,smap3=tfs.tfssim4cg(batch_CTs, batch_CBs, max_vals,filter_size,filter_sigma)
# print('TF 4-c-G custom:')
# print(score3)



# #%%
# score4=tfs.tfmsssim(batch_CTs, batch_CBs, max_vals,filter_size,filter_sigma)
# print('TF MS inbuilt:')
# print(score4)

# score5=tfs.tfmssim_custom(batch_CTs, batch_CBs, max_vals,_MSSSIM_WEIGHTS,filter_size,filter_sigma)
# print('TF MS custom:')
# print(score5)

# score6=tfs.tfmssim_4c(batch_CTs, batch_CBs, max_vals,_MSSSIM_WEIGHTS,filter_size,filter_sigma)
# print('TF MS 4-C custom:')
# print(score6)

# score7=tfs.tfmssim_4cg(batch_CTs, batch_CBs, max_vals,_MSSSIM_WEIGHTS,filter_size,filter_sigma)
# print('TF MS 4-C-G custom:')
# print(score7)

# #%%

# def viewsmap(smap,fignum):
#     smap=np.squeeze(smap,axis=3)
#     smap=np.transpose(smap,(1,2,0))
#     plt.figure(fignum),plt.imshow(smap, cmap='gray'),plt.show()
    
# viewsmap(smap1,6)
# viewsmap(smap2,7)
# viewsmap(smap3,8)

# def gaussian_blur(img, kernel_size=11, sigma=1.5):
#     def gauss_kernel(channels, kernel_size, sigma):
#         ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
#         xx, yy = tf.meshgrid(ax, ax)
#         kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
#         kernel = kernel / tf.reduce_sum(kernel)
#         kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
#         return kernel
#     gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
#     gaussian_kernel = gaussian_kernel[..., tf.newaxis]
#     return tf.nn.conv2d(img, gaussian_kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC')

# y_trues=gaussian_blur(batch_CT, kernel_size=11, sigma=1.5)