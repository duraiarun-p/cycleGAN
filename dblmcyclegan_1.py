#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:40:34 2021

@author: arun
"""

import time
import datetime
import h5py
import numpy as np
from os import listdir
from os import chdir
from os.path import isfile, join
import matplotlib.pyplot as plt
# import ssimmetricslib as ssm
import cv2
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

# import tensorflow_addons as tfa

# from tensorflow.python import keras
# from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
# from tensorflow.keras.layers import UpSampling2D, Conv2D, LeakyReLU, LayerNormalization
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from tensorflow.keras.layers.advanced_activations import LeakyReLU
# from tensorflow.keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()


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
            # d = InstanceNormalization()(d)
            d = LayerNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        return d
    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, stride=1, dropout_rate=0,skip=True):
          """Layers used during upsampling"""
          u = UpSampling2D(size=2)(layer_input)
          u = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same', activation='tanh')(u)
          if dropout_rate:
              u = Dropout(dropout_rate)(u)
          # u = InstanceNormalization()(u)
          u = LayerNormalization()(u)
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
    
    
    
    def __init__(self,mypath,weightoutputpath,epochs,batch_size,imgshape,totalbatchiterations,saveweightflag):
         self.DataPath=mypath
         self.WeightSavePath=weightoutputpath
         self.batch_size=batch_size
         self.img_shape=imgshape
         self.input_shape=tuple([batch_size,imgshape])
         self.genafilter = 32
         self.discfilter = 64
         self.epochs = epochs
         self.totalbatchiterations=totalbatchiterations
         self.lambda_cycle = 10.0                    # Cycle-consistency loss
         self.lambda_id = 0.9 * self.lambda_cycle    # Identity loss
         self.gausssigma=0.5
         self.saveweightflag=saveweightflag
         
         os.chdir(self.WeightSavePath)
         
         optimizer = Adam(0.0002, 0.5)

         self.DiscCT=self.build_discriminator()
         self.DiscCT.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
         self.DiscCT.name='Discriminator-CT'
         # self.DiscCT.summary()
         with open('Disc.txt', 'w+') as f:
             self.DiscCT.summary(print_fn=lambda x: f.write(x + '\n'))
                 
         self.DiscCB=self.build_discriminator()
         self.DiscCB.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
         self.DiscCB.name='Discriminator-CB'
         
         layer_len=len(self.DiscCT.layers)
         layers=self.DiscCT.layers
         labelshapearr=list(layers[layer_len-1].output_shape)
         labelshapearr[0]=self.batch_size
         labelshape=tuple(labelshapearr)
         self.labelshape=labelshape

         self.GenCB2CT=self.build_generator()
         self.GenCB2CT.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
         self.GenCB2CT.name='Generator-CB2CT'
         # self.GenCB2CT.summary()
         with open('Gena.txt', 'w+') as f:
             self.GenCB2CT.summary(print_fn=lambda x: f.write(x + '\n'))
             
         self.GenCT2CB=self.build_generator()
         self.GenCT2CB.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
         self.GenCT2CB.name='Generator-CT2CB'
         
                 # Input images from both domains
         img_CT = Input(shape=self.img_shape)
         img_CB = Input(shape=self.img_shape)

        # Translate images to the other domain
         fake_CB = self.GenCT2CB(img_CT)
         fake_CT = self.GenCB2CT(img_CB)
        # Translate images back to original domain
         reconstr_CT = self.GenCB2CT(fake_CB)
         reconstr_CB = self.GenCT2CB(fake_CT)
        # Identity mapping of images
         img_CT_id = self.GenCT2CB(img_CT)
         img_CB_id = self.GenCB2CT(img_CB)

        # For the combined model we will only train the generators
         self.DiscCT.trainable = False
         self.DiscCB.trainable = False

        # Discriminators determines validity of translated images
         valid_CT = self.DiscCT(fake_CT)
         valid_CB = self.DiscCB(fake_CB)
        
        # SSIM metric as a part of loss function
         # CB1r=cv2.GaussianBlur(img_CT, (7, 7), self.gausssigma)
         # CB2r=cv2.GaussianBlur(img_CB, (7, 7), self.gausssigma)
         
         # CB1r=tfa.image.gaussian_filter2d(img_CT,sigma=self.gausssigma)
        
         # smap,ssim_score=ssm.myfourcompgradssim(CB1r, CB2r)
        
        # Combined model trains generators to fool discriminators
         self.cycleGAN_Model = Model(inputs=[img_CT, img_CB], outputs=[valid_CT, valid_CB, reconstr_CT, reconstr_CB, img_CT_id, img_CB_id])
         self.cycleGAN_Model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id], optimizer=optimizer)
         self.cycleGAN_Model.name='CycleGAN'
         
         with open('cycleGAN.txt', 'w+') as f:
             self.cycleGAN_Model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    def traincgan(self):
        os.chdir(self.WeightSavePath)
        gen1fname="GenCT2CBWeights"
        gen2fname="GenCB2CTWeights"
        disc1fname="DiscCTWeights"
        disc2fname="DiscCBWeights"
        D_losses = np.zeros((self.totalbatchiterations,2,self.epochs))
        G_losses = np.zeros((self.totalbatchiterations,7,self.epochs))
        for epochi in range(self.epochs):
            for batchi in range(self.totalbatchiterations):
                batch_CT, batch_CB=self.dataload()
                valid = np.ones((self.labelshape))
                fake = np.zeros((self.labelshape))
                batch_CT = np.transpose(batch_CT,(2,0,1))
                batch_CT = np.expand_dims(batch_CT, -1)
                batch_CB = np.transpose(batch_CB,(2,0,1))
                batch_CB = np.expand_dims(batch_CB, -1)
                
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_CB = self.GenCT2CB.predict(batch_CT)
                fake_CT = self.GenCB2CT.predict(batch_CB)

                # Train the discriminators (original images = real / translated = Fake)
                dCT_loss_real = self.DiscCT.train_on_batch(batch_CT, valid)
                dCT_loss_fake = self.DiscCT.train_on_batch(fake_CT, fake)
                dCT_loss = 0.5 * np.add(dCT_loss_real, dCT_loss_fake)

                dCB_loss_real = self.DiscCB.train_on_batch(batch_CB, valid)
                dCB_loss_fake = self.DiscCB.train_on_batch(fake_CB, fake)
                dCB_loss = 0.5 * np.add(dCB_loss_real, dCB_loss_fake)

                # Total discriminator loss
                d_loss = 0.5 * np.add(dCT_loss, dCB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.cycleGAN_Model.train_on_batch([batch_CT, batch_CB],
                                                      [valid, valid,
                                                       batch_CT, batch_CB,
                                                       batch_CT, batch_CB])
                D_losses[batchi,:,epochi]=d_loss
                G_losses[batchi,:,epochi]=g_loss
            if epochi % 2 == 0 and self.saveweightflag==True: # Weights saved based on epoch intervals
            
                gen1fname1=gen1fname+'-'+str(epochi)    
                gen2fname1=gen2fname+'-'+str(epochi)
                disc1fname1=disc1fname+'-'+str(epochi)  
                disc2fname1=disc2fname+'-'+str(epochi)
                self.GenCT2CB.save_weights(gen1fname1)
                self.GenCB2CT.save_weights(gen2fname1)
                self.DiscCT.save_weights(disc1fname1)
                self.DiscCB.save_weights(disc2fname1)
            # D1_loss_real = self.Discriminator1.train_on_batch(batch_CT, valid)
            # print('Epoch = %s LD1= %s'%(epochi,D1_loss_real))
        return batch_CT, batch_CB, G_losses, D_losses
        
#%%

mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
weightoutputpath='/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/1392021'
# imgshape=(512,512)

# batch_size=1
# epochs=1
cGAN=CycleGAN(mypath,weightoutputpath,epochs=5,batch_size=5,imgshape=(512,512,1),totalbatchiterations=1,saveweightflag=True)

batch_CT, batch_CB, G_losses, D_losses=cGAN.traincgan()

# TestGenCT2CB=cGAN.build_generator()
# TestGenCT2CB.load_weights("GenCT2CBWeights.h5")
# batch_CB_P=TestGenCT2CB.predict(batch_CT)


#%%  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)