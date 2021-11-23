#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 23:09:24 2021

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
# from sys import getsizeof
# import sys, getopt


# os.environ['CUDA_VISIBLE_DEVICES'] = "" # comment this line when running in eddie
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow import keras
# from tensorflow.keras import backend as K
from tensorflow.keras import layers
# import tensorflow.contrib.eager as tfe
#tf.compat.v1.disable_eager_execution()

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
#%%


class CycleGAN(keras.Model):     

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
        # CTblocksN=(CTsiz1[0]*CTsiz1[1]*CTsiz1[2])//(self.patch_size*self.patch_size*4*self.depth_size)
        # CBblocksN=(CBsiz[0]*CBsiz[1]*CBsiz[2])//(self.patch_size*self.patch_size*4*self.depth_size)
        # i=512//2
        # j=512//2
        # zi=1
        # CTblocks=CT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi:zi+self.depth_size]
        # CBblocks=CBCT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi:zi+self.depth_size]
        # CTblocks=CT[0:32,0:32, 0:32]
        # CBblocks=CBCT[0:32,0:32,0:32]
        # CTblocks=np.expand_dims(CTblocks, axis=0)
        # CBblocks=np.expand_dims(CBblocks, axis=0)
        def blkextraction(CT,CTsiz1,patch_size,depth_size):   
            CTblocks=[]
            for i in  range(0,CTsiz1[0],patch_size*2):
                for j in range(0,CTsiz1[1],patch_size*2):
                    for zi in range(0,CTsiz1[2],depth_size):
                        currentBlk=CT[i:i+patch_size*2,j:j+patch_size*2,zi:zi+depth_size]
                        currentBlk=currentBlk[:,:,0:depth_size]
                        blksiz=currentBlk.shape
                        if blksiz[2] == depth_size:
                            currentBlk=np.expand_dims(currentBlk, axis=0)
                            CTblocks.append(currentBlk)
            return CTblocks             
        CTblocks_all=blkextraction(CT,CTsiz1,self.patch_size,self.depth_size)
        CBblocks_all=blkextraction(CBCT,CBsiz,self.patch_size,self.depth_size)
        CTblocksN=len(CTblocks_all)
        CBblocksN=len(CBblocks_all)
        
        # for blki in range(self.NumberofTrainingSamples):
        #     CTblocks=CTblocks_all[np.random.choice(CTblocksN,replace=False)]
        #     CBblocks=CBblocks_all[np.random.choice(CBblocksN,replace=False)]
            
        # if all scans have been read:
        #    yield None
        # else
            # yield tf.convert_to_tensor(CTblocks, dtype=tf.float32), tf.convert_to_tensor(CBblocks, dtype=tf.float32)
        CTblocks=CTblocks_all[np.random.choice(CTblocksN,replace=False)]
        CBblocks=CBblocks_all[np.random.choice(CBblocksN,replace=False)]
        yield tf.convert_to_tensor(CTblocks, dtype=tf.float32), tf.convert_to_tensor(CBblocks, dtype=tf.float32)
    
    def dataload3D_2(self):
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
        # CTblocksN=(CTsiz1[0]*CTsiz1[1]*CTsiz1[2])//(self.patch_size*self.patch_size*4*self.depth_size)
        # CBblocksN=(CBsiz[0]*CBsiz[1]*CBsiz[2])//(self.patch_size*self.patch_size*4*self.depth_size)
        # i=512//2
        # j=512//2
        # zi=1
        i=np.random.randint(CTsiz1[0]-self.patch_size*2)
        j=np.random.randint(CTsiz1[1]-self.patch_size*2)
        zi1=np.random.randint(CTsiz1[2]-self.depth_size)
        zi2=np.random.randint(CBsiz[2]-self.depth_size)
        CTblocks=CT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi1:zi1+self.depth_size]
        CBblocks=CBCT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi2:zi2+self.depth_size]
        # CTblocks=CT[0:32,0:32, 0:32]
        # CBblocks=CBCT[0:32,0:32,0:32]
        CTblocks=np.expand_dims(CTblocks, axis=0)
        CBblocks=np.expand_dims(CBblocks, axis=0)

        yield tf.convert_to_tensor(CTblocks, dtype=tf.float32), tf.convert_to_tensor(CBblocks, dtype=tf.float32)

    def get_ds(self):
        return tf.data.Dataset.from_generator(
            self.dataload3D_2,
            output_signature=(tf.TensorSpec(shape=(None, 32, 32, 32), dtype=tf.float32), tf.TensorSpec(shape=(None,32, 32, 32), dtype=tf.float32))
            )
    
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
        
    
    def __init__(self,mypath,weightoutputpath,NumberofTrainingSamples,epochs,batch_size,imgshape,patch_size,depth_size,totalbatchiterations,saveweightflag):
         super(CycleGAN,self).__init__()
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
         self.lambda_identity = 0.5 * self.lambda_cycle
         self.saveweightflag=saveweightflag
         self.patch_size=patch_size
         self.input_layer_shape_2D=tuple([1,patch_size*2,patch_size*2,1])
         self.input_layer_shape_3D=tuple([patch_size*2,patch_size*2,depth_size,1])
         self.NumberofTrainingSamples=NumberofTrainingSamples
         
         self.current_learning_rate = 0.003
         # optimizer = keras.optimizers.Adam(self.current_learning_rate)
         
         os.chdir(self.WeightSavePath)
         self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))
         os.mkdir(self.folderlen)
         self.WeightSavePath = os.path.join(self.WeightSavePath, self.folderlen)
         os.chdir(self.WeightSavePath)
         
         newdir='arch'
         
         self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
         os.mkdir(self.WeightSavePathNew)
         os.chdir(self.WeightSavePathNew)
           
         
         self.G=self.build_generator3D(self.input_layer_shape_3D)
         # # self.GenCB2CT3D.summary()
         # # with open('Gena.txt', 'w+') as f:
         # #     self.GenCB2CT3D.summary(print_fn=lambda x: f.write(x + '\n'))
         # self.GenCB2CT3D.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        
         self.X=self.build_discriminator3D(self.input_layer_shape_3D)
         # # self.DiscCT3D.summary()
         # # with open('Disc.txt', 'w+') as f:
         # #     self.DiscCT3D.summary(print_fn=lambda x: f.write(x + '\n'))
         # self.DiscCT3D.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
         
         self.F=self.build_generator3D(self.input_layer_shape_3D)
         # # self.GenCT2CB3D.summary()
         # # with open('Gena.txt', 'w+') as f:
         # #     self.GenCB2CT3D.summary(print_fn=lambda x: f.write(x + '\n'))
         # self.GenCT2CB3D.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        
         self.Y=self.build_discriminator3D(self.input_layer_shape_3D)
         # # self.DiscCB3D.summary()
         # # with open('Disc.txt', 'w+') as f:
         # #     self.DiscCT3D.summary(print_fn=lambda x: f.write(x + '\n'))
         # self.DiscCB3D.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
         self.mycompile()
    
    

# Define the loss function for the generator
    
    def generator_loss_fnx(self, fake):
        adv_loss_fn = keras.losses.MeanSquaredError()
        fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
        return fake_loss
    
    
    # Define the loss function for the discriminators
    def discriminator_loss_fnx(self, real, fake):
        adv_loss_fn = keras.losses.MeanSquaredError()
        real_loss = adv_loss_fn(tf.ones_like(real), real)
        fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
        return (real_loss + fake_loss) * 0.5
         
    def mycompile(self):
        super(CycleGAN,self).compile()
        self.G_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.F_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.X_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.Y_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.generator_loss_fn = self.generator_loss_fnx
        self.discriminator_loss_fn = self.discriminator_loss_fnx
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
    
    def train_step(self, real_x, real_y):
# real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:
     
 # Horse to fake zebra
            fake_y = self.G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.F(real_y, training=True)
            
            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.G(fake_x, training=True)
            
            # Identity mapping
            same_x = self.F(real_x, training=True)
            same_y = self.G(real_y, training=True)
            
            # Discriminator output
            disc_real_x = self.X(real_x, training=True)
            disc_fake_x = self.X(fake_x, training=True)
            
            disc_real_y = self.Y(real_y, training=True)
            disc_fake_y = self.Y(fake_y, training=True)
            
            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)
            
            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle
            
            # Generator identity loss
            id_loss_G = (self.identity_loss_fn(real_y, same_y) * self.lambda_cycle * self.lambda_identity)
            id_loss_F = (self.identity_loss_fn(real_x, same_x) * self.lambda_cycle * self.lambda_identity)
            
            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F
            
            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)
            
            # Get the gradients for the generators
            grads_G = tape.gradient(total_loss_G, self.G.trainable_variables)
            grads_F = tape.gradient(total_loss_F, self.F.trainable_variables)
            
            # Get the gradients for the discriminators
            disc_X_grads = tape.gradient(disc_X_loss, self.X.trainable_variables)
            disc_Y_grads = tape.gradient(disc_Y_loss, self.Y.trainable_variables)
            
            # Update the weights of the generators
            self.G_optimizer.apply_gradients(zip(grads_G, self.G.trainable_variables))
            self.F_optimizer.apply_gradients(zip(grads_F, self.F.trainable_variables))
            
            # Update the weights of the discriminators
            self.X_optimizer.apply_gradients(zip(disc_X_grads, self.X.trainable_variables))
            self.Y_optimizer.apply_gradients(zip(disc_Y_grads, self.Y.trainable_variables))
            
            return {"G_loss": total_loss_G,"F_loss": total_loss_F,"D_X_loss": disc_X_loss,"D_Y_loss": disc_Y_loss}
    

    
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


cGAN=CycleGAN(mypath,weightoutputpath,NumberofTrainingSamples=1000,epochs=1,batch_size=3,imgshape=(512,512,1),patch_size=16,depth_size=32,totalbatchiterations=2,saveweightflag=False)

# batch_CT_img_patch, batch_CB_img_patch =cGAN.dataload2D()
# batch_CT_img_patch_3d, batch_CB_img_patch_3d =cGAN.dataload3D()
# CTblks,CBblks=cGAN.dataload3D_1()


batch_size = 250
buffer_size = 10*batch_size

train_ds = cGAN.get_ds()

# cGAN.dataload3D_1()
# train_ds = train_ds.batch(batch_size)
# train_ds = train_ds.repeat(count=2)
# train_ds = train_ds.shuffle(buffer_size)
# train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# train_ds1=list(train_ds.repeat(batch_size))

epochs=10
# for epoch in range(epochs):
#     train_ds=cGAN.dataload3D_1()
    # for batchi in range(batch_size):
        # for x, y in train_ds:
        # train_ds = cGAN.get_ds()
        # x,y=next(iter(train_ds))
        # train_results = cGAN.train_step(real_x=x, real_y=y)
        # print('Current batch: %s'%batchi)
            # print(train_results)
        # break

# epochs=1
# batch_size = 10
# for epoch in range(epochs):
#     # train_ds=cGAN.dataload3D_1()
#     train_ds1=list(train_ds.repeat(batch_size))
#     for batchi in range(batch_size):
#         x,y=train_ds1[batchi][0],train_ds1[batchi][1]
#         # y=train_ds1[batchi][1]
#         train_results = cGAN.train_step(real_x=x, real_y=y)
        # print('Current batch: %s'%batchi)
        
# This sub-routine works well and checks the spec of dataset        
# for batch_index, (x,y) in enumerate(train_ds):
#   pass
# print("batch: ", batch_index)
# print("Data shape: ", x.shape, y.shape)

# epochs=1
# for epoch in range(epochs):
#     for batch_index, (x,y) in enumerate(train_ds):
#         for _ in range(len)
#         x,y=next(iter(train_ds))
#         train_results = cGAN.train_step(real_x=x, real_y=y)
#         print(train_results)
#         print('Current batch: %s'%batch_index)

for epoch in range(epochs):
    for batch_index in range(batch_size):
        x,y=next(iter(train_ds))
        train_results = cGAN.train_step(real_x=x, real_y=y)
        # print(train_results)
        print('Current batch: %s'%batch_index)
        
    



#%%  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)