#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:24:21 2021

@author: arun
"""
import time
import datetime
# import cv2
# import itk
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
import os
from sys import getsizeof
# import sys, getopt


os.environ['CUDA_VISIBLE_DEVICES'] = "" # comment this line when running in eddie
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
# import tensorflow.contrib.eager as tfe
tf.compat.v1.enable_eager_execution()

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
#%%


class CycleGAN():     

    def printmypath(self):
        print(self.DataPath)
        
    def dataload2D(self):
        self.batch_size_2d=1
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
        CT_rand_index=np.random.choice(CTsiz1[2],size=self.batch_size_2d,replace=False)
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
        batch_CB_img=np.zeros((CTsiz1[0],CTsiz1[1],self.batch_size_2d))
        for cbi in range(self.batch_size_2d):
            CB_rand_sl_index=np.random.choice(CBsiz[2])
            CB_rand_pat_index=np.random.choice(CBCLen[1],replace=False)
            # print(CB_rand_pat_index)
            # print(CB_rand_sl_index)
            batch_CB_img[:,:,cbi]=CBCTs[CB_rand_pat_index][:,:,CB_rand_sl_index]
        del mat_contents
        batch_CT_img_patch=batch_CT_img[CTsiz1[0]//2-self.patch_size:CTsiz1[0]//2+self.patch_size,CTsiz1[1]//2-self.patch_size:CTsiz1[1]//2+self.patch_size,:]
        batch_CB_img_patch=batch_CB_img[CBsiz[0]//2-self.patch_size:CBsiz[0]//2+self.patch_size,CBsiz[1]//2-self.patch_size:CBsiz[1]//2+self.patch_size,:]
        return batch_CT_img_patch, batch_CB_img_patch
    
    def dataload3D(self):
        # self.batch_size=1
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
        CT_rand_index=np.random.choice(CTsiz1[2],size=self.depth_size,replace=False)
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
        batch_CB_img=np.zeros((CTsiz1[0],CTsiz1[1],self.depth_size))
        for cbi in range(self.depth_size):
            CB_rand_sl_index=np.random.choice(CBsiz[2])
            CB_rand_pat_index=np.random.choice(CBCLen[1],replace=False)
            # print(CB_rand_pat_index)
            # print(CB_rand_sl_index)
            batch_CB_img[:,:,cbi]=CBCTs[CB_rand_pat_index][:,:,CB_rand_sl_index]
        del mat_contents
        batch_CT_img_patch=batch_CT_img[CTsiz1[0]//2-self.patch_size:CTsiz1[0]//2+self.patch_size,CTsiz1[1]//2-self.patch_size:CTsiz1[1]//2+self.patch_size,:]
        batch_CB_img_patch=batch_CB_img[CBsiz[0]//2-self.patch_size:CBsiz[0]//2+self.patch_size,CBsiz[1]//2-self.patch_size:CBsiz[1]//2+self.patch_size,:]
        return batch_CT_img_patch, batch_CB_img_patch
    
    def dataload3D_1(self):
        # self.batch_size=1
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
        CT=np.transpose(CT,(2,1,0))#data volume
        CTsiz1=CT.shape  
        CBCTCellRef=mat_contents['CBCTInfocell']
        CBCLen=np.shape(CBCTCellRef)
        # CBCTs=[]
        CBCTi=np.random.choice(CBCLen[1],replace=False)  
        CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
        CBCellRef=mat_contents[CBCellRef]
        CBCT=CBCellRef[()]
        CBCT=np.transpose(CBCT,(2,1,0))
        CBsiz=CBCT.shape
        CTblocksN=(CTsiz1[0]*CTsiz1[1]*CTsiz1[2])//(self.patch_size*self.patch_size*4*self.depth_size)
        CBblocksN=(CBsiz[0]*CBsiz[1]*CBsiz[2])//(self.patch_size*self.patch_size*4*self.depth_size)
        def blkextraction(CT,CTsiz1,patch_size,depth_size):   
            CTblocks=[]
            for i in  range(0,CTsiz1[0],patch_size*2):
                for j in range(0,CTsiz1[1],patch_size*2):
                    for zi in range(0,CTsiz1[2],depth_size):
                        currentBlk=CT[i:i+patch_size*2,j:j+patch_size*2,zi:zi+depth_size]
                        currentBlk=currentBlk[:,:,0:depth_size]
                        blksiz=currentBlk.shape
                        if blksiz[2] == depth_size:
                            CTblocks.append(currentBlk)
            return CTblocks             
        trainCT=blkextraction(CT,CTsiz1,self.patch_size,self.depth_size)
        trainCB=blkextraction(CBCT,CBsiz,self.patch_size,self.depth_size)
        # trainCT=tf.data.Dataset.from_tensor_slices(CTblocks)
        # trainCB=tf.data.Dataset.from_tensor_slices(CBblocks)
        return trainCT,trainCB
    
#%% 2D     
    def convblk2d(ipL,filters,k_size,step_size):
        opL=layers.Conv2D(filters, kernel_size=k_size, strides=step_size,padding='SAME')(ipL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.LeakyReLU()(opL)
        return opL
        
        
    def build_generator2D(self,input_layer_shape):
        ipL=keras.Input(shape=input_layer_shape,name='Input')
        opL=self.convblk2d(ipL,filters=2,k_size=2,step_size=1)
        
        return keras.Model(ipL,opL)
#%% 3D
    @staticmethod
    def convblk3d(ipL,filters,kernel_size,strides):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.LeakyReLU()(opL)
        return opL
    
    @staticmethod
    def attentionblk3D(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=3)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def attentionblk3D_1(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=13)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def attentionblk3D_2(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=25)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def deconvblk3D(ipL,filters,kernel_size,strides):
        opL=layers.UpSampling3D(size=2)(ipL)
        opL=layers.Conv3D(filters, kernel_size=1, strides=strides,padding='SAME')(opL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.LeakyReLU()(opL)
        return opL
    
    @staticmethod
    def resblock(x,filters,kernelsize):
        fx = layers.Conv3D(filters, kernelsize, activation='relu', padding='same')(x)
        fx = layers.BatchNormalization()(fx)
        fx = layers.Conv3D(filters, kernelsize, padding='same')(fx)
        out = layers.Add()([x,fx])
        out = layers.ReLU()(out)
        out = layers.BatchNormalization()(out)
        return out
    
    def build_generator3D(self,input_layer_shape):
        ipL=keras.Input(shape=input_layer_shape,name='Input')
        opL1=self.convblk3d(ipL,self.genafilter,self.kernel_size,self.stride2)
        opL2=self.convblk3d(opL1,self.genafilter*2,self.kernel_size,self.stride2)
        opL3=self.convblk3d(opL2,self.genafilter*4,self.kernel_size,self.stride2)
        opL4=layers.Conv3D(filters=self.genafilter*8, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL3)
        # opL4=self.convblk3d(opL3,self.genafilter*8,self.kernel_size,self.stride2)
        
        opL5=self.attentionblk3D(opL3,opL4,filters=self.genafilter*8, kernel_size=self.kernel_size, strides=self.stride2)
        
        opL6=layers.Concatenate()([opL4,opL5])
        opL7=self.deconvblk3D(opL6,self.genafilter*4,self.kernel_size,self.stride1)
        opL8=self.deconvblk3D(opL7,self.genafilter*2,self.kernel_size,self.stride1)
        # opL9=self.convblk3d(opL8,self.genafilter*4,self.kernel_size,self.stride2)
        opL9=layers.Conv3D(filters=self.genafilter*4, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL8)
        
        opL10=self.attentionblk3D_1(opL1,opL9,filters=self.genafilter*4, kernel_size=self.kernel_size, strides=self.stride2)
        
        opL11=layers.Concatenate()([opL9,opL10])
        opL12=self.deconvblk3D(opL11,self.genafilter*2,self.kernel_size,self.stride1)
        opL13=self.deconvblk3D(opL12,self.genafilter*4,self.kernel_size,self.stride1)
        # opL14=self.convblk3d(opL13,self.genafilter*2,self.kernel_size,self.stride2)
        opL14=layers.Conv3D(filters=self.genafilter*2, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL13)
        
        opL15=self.attentionblk3D_2(ipL,opL14,filters=self.genafilter*2, kernel_size=self.kernel_size, strides=self.stride2)
        
        opL16=layers.Concatenate()([opL14,opL15])
        opL17=self.deconvblk3D(opL16,self.genafilter*1,self.kernel_size,self.stride1)
        opL18=self.deconvblk3D(opL17,self.genafilter*2,self.kernel_size,self.stride1)
        # opL19=self.convblk3d(opL18,self.genafilter*1,self.kernel_size,self.stride1)
        opL19=layers.Conv3D(filters=self.genafilter*1, kernel_size=self.kernel_size, strides=self.stride1,padding='SAME')(opL18)
        
        opL20=self.resblock(opL19, self.genafilter*1, self.kernel_size)
        opL21=self.resblock(opL20, self.genafilter*1, self.kernel_size)
        opL22=layers.Conv3D(filters=1, kernel_size=self.kernel_size, strides=self.stride1,padding='SAME')(opL21)
        
        return keras.Model(ipL,opL22)
    
    def build_discriminator3D(self,input_layer_shape):
        ipL=keras.Input(shape=input_layer_shape,name='Input')
        opL1=self.convblk3d(ipL,self.discfilter,self.kernel_size_disc,self.stride2)
        opL2=self.convblk3d(opL1,self.discfilter*2,self.kernel_size_disc,self.stride2)
        opL3=self.convblk3d(opL2,self.discfilter*4,self.kernel_size_disc,self.stride2)
        opL4=self.convblk3d(opL3,self.discfilter*8,self.kernel_size_disc,self.stride1)
        opL5=layers.Dense(self.discfilter*16)(opL4)
        opL5=layers.LeakyReLU()(opL5)
        opL6=layers.Dense(self.discfilter*8)(opL5)
        opL6=layers.LeakyReLU()(opL6)
        opL7=layers.Dense(1)(opL6)
        opL7=layers.Activation('sigmoid')(opL7)
        
        return keras.Model(ipL,opL7)
        
    
    def __init__(self,mypath,weightoutputpath,epochs,batch_size,patch_size,depth_size,imgshape,totalbatchiterations,saveweightflag):
         self.DataPath=mypath
         self.WeightSavePath=weightoutputpath
         self.batch_size=batch_size
         self.depth_size=depth_size
         self.img_shape=imgshape
         # self.input_shape=tuple([batch_size,imgshape])
         self.genafilter = 16
         self.discfilter = 16
         self.kernel_size = 3
         self.kernel_size_disc = 4
         self.stride2=2
         self.stride1=1
         self.epochs = epochs
         self.totalbatchiterations=totalbatchiterations
         self.lambda_cycle = 10.0                    # Cycle-consistency loss
         self.lambda_id = 0.9 * self.lambda_cycle    # Identity loss
         self.saveweightflag=saveweightflag
         self.patch_size=patch_size
         self.input_layer_shape_2D=tuple([1,patch_size*2,patch_size*2,1])
         self.input_layer_shape_3D=tuple([patch_size*2,patch_size*2,depth_size,1])
         
         os.chdir(self.WeightSavePath)
         self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))
         os.mkdir(self.folderlen)
         self.WeightSavePath = os.path.join(self.WeightSavePath, self.folderlen)
         os.chdir(self.WeightSavePath)
         
         newdir='arch'
         
         self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
         os.mkdir(self.WeightSavePathNew)
         os.chdir(self.WeightSavePathNew)
         
         self.current_learning_rate = 0.003
         optimizer = keras.optimizers.Adam(self.current_learning_rate)

         self.DiscCT=self.build_discriminator3D(self.input_layer_shape_3D)
         self.DiscCT.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
         self.DiscCT._name='Discriminator-CT'
         # self.DiscCT.summary()
         with open('Disc.txt', 'w+') as f:
             self.DiscCT.summary(print_fn=lambda x: f.write(x + '\n'))
                 
         self.DiscCB=self.build_discriminator3D(self.input_layer_shape_3D)
         self.DiscCB.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
         self.DiscCB._name='Discriminator-CB'
         
         layer_len=len(self.DiscCT.layers)
         layers_N=self.DiscCT.layers
         labelshapearr=list(layers_N[-1].output_shape)
         # labelshapearr[0]=self.depth_size
         labelshape=tuple(labelshapearr)
         self.labelshape=labelshape[1:5]

         self.GenCB2CT=self.build_generator3D(self.input_layer_shape_3D)
         self.GenCB2CT.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
         self.GenCB2CT._name='Generator-CB2CT'
         # self.GenCB2CT.summary()
         with open('Gena.txt', 'w+') as f:
             self.GenCB2CT.summary(print_fn=lambda x: f.write(x + '\n'))
             
         self.GenCT2CB=self.build_generator3D(self.input_layer_shape_3D)
         self.GenCT2CB.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
         self.GenCT2CB._name='Generator-CT2CB'
         
                 # Input images from both domains
         img_CT = keras.Input(shape=self.input_layer_shape_3D)
         img_CB = keras.Input(shape=self.input_layer_shape_3D)

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

        # Combined model trains generators to fool discriminators
         self.cycleGAN_Model = keras.Model(inputs=[img_CT, img_CB], outputs=[valid_CT, valid_CB, reconstr_CT, reconstr_CB, img_CT_id, img_CB_id])
         self.cycleGAN_Model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id], optimizer=optimizer)
         self.cycleGAN_Model._name='CycleGAN'
         
         with open('cycleGAN.txt', 'w+') as f:
             self.cycleGAN_Model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    def traincgan(self):
        os.chdir(self.WeightSavePathNew)

         # self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))
         
        newdir='weights'
         
        self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
        os.mkdir(self.WeightSavePathNew)
        os.chdir(self.WeightSavePathNew)
        
        gen1fname="GenCT2CBWeights"
        gen2fname="GenCB2CTWeights"
        disc1fname="DiscCTWeights"
        disc2fname="DiscCBWeights"
        D_losses = np.zeros((self.totalbatchiterations,2,self.epochs))
        G_losses = np.zeros((self.totalbatchiterations,7,self.epochs))
        for epochi in range(self.epochs):
            batch_CTblks, batch_CBblks=self.dataload3D_1()
            CBL=len(batch_CBblks)
            CTL=len(batch_CTblks)
            for batchi in range(self.totalbatchiterations):
                batch_CT=batch_CTblks[np.random.choice(CTL,replace=False)]
                batch_CB=batch_CBblks[np.random.choice(CBL,replace=False)]
                valid = np.ones((self.labelshape))
                fake = np.zeros((self.labelshape))
                # batch_CT = np.transpose(batch_CT,(2,0,1))
                batch_CT = np.expand_dims(batch_CT, 0)
                # batch_CB = np.transpose(batch_CB,(2,0,1))
                batch_CB = np.expand_dims(batch_CB, 0)
                valid = np.expand_dims(valid, 0)
                fake = np.expand_dims(fake, 0)
                
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
                # self.current_learning_rate  = self.current_learning_rate / 2
                K.set_value(self.cycleGAN_Model.optimizer.learning_rate, self.current_learning_rate)
                # Train the generators
                g_loss = self.cycleGAN_Model.train_on_batch([batch_CT, batch_CB],
                                                      [valid, valid,
                                                       batch_CT, batch_CB,
                                                       batch_CT, batch_CB])
                D_losses[batchi,:,epochi]=d_loss
                G_losses[batchi,:,epochi]=g_loss
            if epochi % 1 == 0 and self.saveweightflag==True: # Weights saved based on epoch intervals
                gen1fname1=gen1fname+'-'+str(epochi)+'.h5'    
                gen2fname1=gen2fname+'-'+str(epochi)+'.h5'
                disc1fname1=disc1fname+'-'+str(epochi)+'.h5'
                disc2fname1=disc2fname+'-'+str(epochi)+'.h5'
                self.GenCT2CB.save_weights(gen1fname1)
                self.GenCB2CT.save_weights(gen2fname1)
                self.DiscCT.save_weights(disc1fname1)
                self.DiscCB.save_weights(disc2fname1)
        return batch_CT, batch_CB, G_losses, D_losses
    def predictCT2CB(self,batch_CT,batch_CB,GenCT2CBWeightsFileName):
        batch_CT = np.expand_dims(batch_CT, 0)
        batch_CB = np.expand_dims(batch_CB, 0)
        # GenCT2CBWeightsFileName="GenCT2CBWeights.h5"
        TestGenCT2CB=self.build_generator3D(self.input_layer_shape_3D)
        TestGenCT2CB.load_weights(GenCT2CBWeightsFileName)
        batch_CB_P=TestGenCT2CB.predict(batch_CT)
        batch_CB_P=np.squeeze(batch_CB_P,axis=-1)
        batch_CB_P=np.squeeze(batch_CB_P,axis=0)
        return batch_CB_P
        
#%%

mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
outputpath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/output'
weightoutputpath='/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/19102021/alpha'
# imgshape=(512,512)

# inputfile = ''
# outputfile = ''
# try:
#   argv=sys.argv[1:]
#   opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
# except getopt.GetoptError:
#   print(' Check syntax: test.py -i <inputfile> -o <outputfile>')
#   sys.exit(2)
# for opt, arg in opts:
#   if opt == '-h':
#       print('test.py -i <inputfile> -o <outputfile>')
#       sys.exit()
#   elif opt in ("-i", "--ifile"):
#       mypath = arg
#   elif opt in ("-o", "--ofile"):
#       weightoutputpath = arg
# print('Input path is :', mypath)
# print('Output path is :', weightoutputpath)

# batch_size=1
# epochs=1
cGAN=CycleGAN(mypath,weightoutputpath,epochs=5,batch_size=3,patch_size=16,depth_size=32,imgshape=(512,512,1),totalbatchiterations=200,saveweightflag=True)

tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=tfconfig)
tf.compat.v1.keras.backend.set_session(sess)

# batch_CT, batch_CB, G_losses, D_losses=cGAN.traincgan()


#%%
batch_CT, batch_CB =cGAN.dataload3D_1()
GenCT2CBWeightsFileName="/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/19102021/alpha/run0/weights/GenCT2CBWeights-4.h5"
batch_CB_P=cGAN.predictCT2CB(batch_CT[100],batch_CB[100],GenCT2CBWeightsFileName)
#%%
# TestGenCT2CB=cGAN.build_generator()
# TestGenCT2CB.load_weights("/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/weights/GenCT2CBWeights-200.h5")
# batch_CB_P=TestGenCT2CB.predict(batch_CT)
#%%
# from scipy.io import savemat
# mdic = {"batch_CT":batch_CT,"batch_CB_P":batch_CB_P,"batch_CT":batch_CT}
# savemat("Testmat.mat",mdic)

#%%  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)