#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:00:18 2021

@author: arun
"""
import time
import datetime
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import keras
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

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
    
    @staticmethod
    def conv2d(layer_input, filters, f_size=4,stride=2,normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size,strides=stride, padding='same',activation='relu')(layer_input)
        if normalization:
            d = InstanceNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        return d
    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, stride=1, dropout_rate=0,skip=True):
          """Layers used during upsampling"""
          u = UpSampling2D(size=2)(layer_input)
          u = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same', activation='tanh')(u)
          if dropout_rate:
              u = Dropout(dropout_rate)(u)
          u = InstanceNormalization()(u)
          if skip:
              u = Concatenate()([u, skip_input])
          return u
    
    def build_discriminator(self):
        d0 = Input(shape=self.img_shape,name='Input')
        d1 = self.conv2d(d0, self.discfilter, stride=2, normalization=False)
        d2 = self.conv2d(d1, self.discfilter*2, stride=2, normalization=True)
        d3 = self.conv2d(d2, self.discfilter*4, stride=2, normalization=True)
        d4 = self.conv2d(d3, self.discfilter*8, stride=2, normalization=True)
        d5 = self.conv2d(d4, self.discfilter*8, stride=1, normalization=True)
        d6 = Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)
        
        return Model(d0,d6)
    
    def build_generator(self):
        d0 = Input(shape=self.img_shape,name='Input')
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
        u1 = Conv2D(1, kernel_size=1, strides=1, padding='same')(u2)
        
        return Model(d0,u1)
    
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
    
     optimizer = Adam(0.0002, 0.5)

#%%
mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
cGAN=CycleGAN(mypath,epochs=2,batch_size=5,imgshape=(512,512,1),totalbatchiterations=400)

batch_CT, batch_CB =cGAN.dataload()
GenCT2CBWeightsFileName="/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/weights/GenCT2CBWeights-200.h5"
batch_CB_P=cGAN.predictCT2CB(batch_CT,batch_CB,GenCT2CBWeightsFileName)

# from scipy.io import savemat
# mdic = {"batch_CB":batch_CB,"batch_CB_P":batch_CB_P,"batch_CT":batch_CT}
# savemat("Testmat.mat",mdic)

# import itk
# ITEimg1="CBP"+".nrrd"
# IT1img1="CT"+".nrrd"
# ITEimg=itk.GetImageFromArray(batch_CB_P)
# itk.imwrite(ITEimg, ITEimg1)
# IT1img=itk.GetImageFromArray(batch_CT)
# itk.imwrite(IT1img, IT1img1)

#%%  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)