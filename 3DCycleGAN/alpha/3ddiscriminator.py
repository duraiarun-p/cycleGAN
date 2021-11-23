#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 22:30:04 2021

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
from scipy import signal as sg
from scipy.io import loadmat
from scipy.io import savemat


os.environ['CUDA_VISIBLE_DEVICES'] = "" # comment this line when running in eddie
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow import keras
# from tensorflow.keras import backend as K
from tensorflow.keras import layers
# import tensorflow.contrib.eager as tfe
#tf.compat.v1.disable_eager_execution()

# cfg = tf.compat.v1.ConfigProto() 
# cfg.gpu_options.allow_growth = True
# sess= tf.compat.v1.Session(config=cfg)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
print('Script started at')
print(st_0)
start_time_0=time.time()
#%%
class CycleGAN(keras.Model):
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
        # CT=(CT-np.min(CT))/np.ptp(CT)
        CTsiz1=CT.shape  
        CBCTCellRef=mat_contents['CBCTInfocell']
        CBCLen=np.shape(CBCTCellRef)
        CBCTi=np.random.choice(CBCLen[1],replace=False)  
        CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
        CBCellRef=mat_contents[CBCellRef]
        CBCT=CBCellRef[()]
        CBCT=np.transpose(CBCT,(2,1,0))
        # CBCT=(CBCT-np.min(CBCT))/np.ptp(CBCT)
        CBsiz=CBCT.shape
        i=np.random.randint(CTsiz1[0]-self.patch_size*2)
        j=np.random.randint(CTsiz1[1]-self.patch_size*2)
        zi1=np.random.randint(CTsiz1[2]-self.depth_size)
        zi2=np.random.randint(CBsiz[2]-self.depth_size)
        CTblocks=CT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi1:zi1+self.depth_size]
        CBblocks=CBCT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi2:zi2+self.depth_size]
        # CBblocks=np.random.randint(100, size=(self.patch_size*2, self.patch_size*2,self.depth_size))
        CTblocks=np.expand_dims(CTblocks, axis=0)
        CBblocks=np.expand_dims(CBblocks, axis=0)

        yield tf.convert_to_tensor(CTblocks, dtype=tf.float32), tf.convert_to_tensor(CBblocks, dtype=tf.float32)
        
    def dataload3D_2_test(self):
        # self.batch_size=1
        # mypath=self.DataPath
        onlyfiles = [f for f in listdir(self.DataPath) if isfile(join(self.DataPath, f))]
        onlyfiles.sort()
        onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
        onlyfiles = onlyfiles[-onlyfileslenrem:]
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
        # CT=(CT-np.min(CT))/np.ptp(CT)
        CTsiz1=CT.shape  
        CBCTCellRef=mat_contents['CBCTInfocell']
        CBCLen=np.shape(CBCTCellRef)
        CBCTi=np.random.choice(CBCLen[1],replace=False)  
        CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
        CBCellRef=mat_contents[CBCellRef]
        CBCT=CBCellRef[()]
        CBCT=np.transpose(CBCT,(2,1,0))
        # CBCT=(CBCT-np.min(CBCT))/np.ptp(CBCT)
        CBsiz=CBCT.shape
        i=np.random.randint(CTsiz1[0]-self.patch_size*2)
        j=np.random.randint(CTsiz1[1]-self.patch_size*2)
        zi1=np.random.randint(CTsiz1[2]-self.depth_size)
        zi2=np.random.randint(CBsiz[2]-self.depth_size)
        CTblocks=CT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi1:zi1+self.depth_size]
        # CBblocks=CBCT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi2:zi2+self.depth_size]
        CBblocks=np.random.randint(10, size=(self.patch_size*2, self.patch_size*2,self.depth_size))
        CTblocks=np.expand_dims(CTblocks, axis=0)
        CBblocks=np.expand_dims(CBblocks, axis=0)

        yield tf.convert_to_tensor(CTblocks, dtype=tf.float32), tf.convert_to_tensor(CBblocks, dtype=tf.float32)

    def get_ds(self):
        return tf.data.Dataset.from_generator(
            self.dataload3D_2,
            output_signature=(tf.TensorSpec(shape=(None, 32, 32, 32), dtype=tf.float32), tf.TensorSpec(shape=(None,32, 32, 32), dtype=tf.float32))
            )
    
    def get_ds_test(self):
        return tf.data.Dataset.from_generator(
            self.dataload3D_2_test,
            output_signature=(tf.TensorSpec(shape=(None, 32, 32, 32), dtype=tf.float32), tf.TensorSpec(shape=(None,32, 32, 32), dtype=tf.float32))
            )
    
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
     
    def build_discriminator3D(self,input_layer_shape):
        ipL=keras.Input(shape=input_layer_shape,name='Input')
        opL1=self.convblk3d(ipL,self.discfilter,self.kernel_size_disc,self.stride2)
        opL2=self.convblk3d(opL1,self.discfilter*2,self.kernel_size_disc,self.stride2)
        opL3=self.convblk3d(opL2,self.discfilter*4,self.kernel_size_disc,self.stride2)
        opL4=self.convblk3d(opL3,self.discfilter*8,self.kernel_size_disc,self.stride1)
        
        
        opL5=layers.Flatten()(opL4)
        
        opL5=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL5)
        opL5=layers.LeakyReLU()(opL5)
        
        # opL6=layers.Flatten()(opL5)
        opL6=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*16)(opL5)
        opL6=layers.LeakyReLU()(opL6)
        
        # opL7=layers.Flatten()(opL6)
        opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL6)
        opL8=layers.Activation('sigmoid')(opL7)
        
        # opL9=layers.Flatten()(opL8)
        opL9=layers.Dense(1)(opL8)
        opL9=layers.Activation('sigmoid')(opL9)
        
        return keras.Model(ipL,opL9)
    
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
         
         self.current_learning_rate = 0.0002
         self.learning_rate = 0.0002
         
         os.chdir(self.WeightSavePath)
         self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))
         os.mkdir(self.folderlen)
         self.WeightSavePath = os.path.join(self.WeightSavePath, self.folderlen)
         os.chdir(self.WeightSavePath)
         
         newdir='arch'
         
         self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
         os.mkdir(self.WeightSavePathNew)
         os.chdir(self.WeightSavePathNew)
         self.X=self.build_discriminator3D(self.input_layer_shape_3D)
         # with open('Disc.txt', 'w+') as f:
         #     self.X.summary(print_fn=lambda x: f.write(x + '\n'))
         # self.X.summary()
         self.mycompile_1()
         self.checkpoint = tf.train.Checkpoint(X=self.X,X_optimizer=self.X_optimizer)         
         self.checkpoint_dir = os.path.join(self.WeightSavePath, 'checkpoints')
         if not os.path.exists(self.checkpoint_dir):
             os.makedirs(self.checkpoint_dir)
         os.chdir(self.checkpoint_dir)
    
    # Define the loss function for the discriminators
    # Doesnt look like a loss function suitable for discriminator
    def discriminator_loss_fnx(self, real, fake):
        adv_loss_fn = keras.losses.BinaryCrossentropy()
        real_loss = adv_loss_fn(tf.ones_like(real), real)
        fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
        return (real_loss + fake_loss) * 1
    
    def mycompile_1(self):
        super(CycleGAN,self).compile()
        self.X_optimizer = keras.optimizers.Adam(self.current_learning_rate, beta_1=0.5)
        self.discriminator_loss_fn = self.discriminator_loss_fnx
    # @tf.function    
    def train_step(self,real_x, fake_x):
        with tf.GradientTape(persistent=True) as tape:
            # Discriminator output
            disc_real_x = self.X(real_x, training=True)
            disc_fake_x = self.X(fake_x, training=True)
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_X_grads = tape.gradient(disc_X_loss, self.X.trainable_variables)
            self.X_optimizer.apply_gradients(zip(disc_X_grads, self.X.trainable_variables))
        return disc_X_loss.numpy()
    # @tf.function
    def test_step(self,real_x, fake_x):
        disc_real_x = self.X(real_x, training=False)
        disc_fake_x = self.X(fake_x, training=False)
        disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
        return disc_X_loss.numpy()
    
    def train_network(self):
        trainresult=[]
        testresult=[]
        for epoch in range(self.epochs):
            train_ds=self.get_ds()
            test_ds = self.get_ds_test()
            # self.current_learning_rate=self.learning_rate
            for batch_index in range(self.totalbatchiterations):
                x,y=next(iter(train_ds))
                test_x,test_y=next(iter(test_ds))
                train_results = self.train_step(real_x=x, fake_x=y)
                test_results = self.test_step(real_x=test_x, fake_x=test_y)
        
            trainresult.append(train_results)
            testresult.append(test_results)
            print('Epoch: %s'%epoch)
        return trainresult,testresult
    
    def learningrate_log_scheduler(self):
        learning_rates=np.logspace(-8, 0,num=self.epochs)
        return learning_rates
    
    def learning_rate_identifier(self):
        trainresult=[]
        testresult=[]
        learning_rates=self.learningrate_log_scheduler()
        for epoch in range(self.epochs):
            train_ds=self.get_ds()
            test_ds = self.get_ds_test()
            self.current_learning_rate=learning_rates[epoch]
            # self.X_optimizer.learning_rate.assign(self.current_learning_rate)
            for batch_index in range(self.totalbatchiterations):
                x,y=next(iter(train_ds))
                test_x,test_y=next(iter(test_ds))
                train_results = self.train_step(real_x=x, fake_x=y)
            
            test_results = self.test_step(real_x=test_x, fake_x=test_y)
        
            trainresult.append(train_results)
            testresult.append(test_results)
            print('Epoch: %s'%epoch)
        return trainresult,testresult
    
    def learning_rate_identifier_batch(self):
        trainresult=[]
        testresult=[]
        learning_rates=self.learningrate_log_scheduler()

        for epoch in range(self.epochs):
            train_ds=self.get_ds()
            test_ds = self.get_ds_test()
            self.current_learning_rate=learning_rates[epoch]
            disc_X_loss_lis=tf.constant(0,dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
            # disc_X_loss_lis=[]
            # self.X_optimizer.learning_rate.assign(self.current_learning_rate)
                for batch_index in range(self.totalbatchiterations):
                    real_x,fake_x=next(iter(train_ds))
                # Discriminator output
                    disc_real_x = self.X(real_x, training=True)
                    disc_fake_x = self.X(fake_x, training=True)
                    disc_X_loss_ele = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
                    disc_X_loss_lis = tf.add(disc_X_loss_lis,disc_X_loss_ele)
                    # disc_X_loss_lis.append(disc_X_loss_ele)
            # disc_X_loss=tf.reduce_mean(disc_X_loss_lis)
                disc_X_loss=disc_X_loss_lis/self.totalbatchiterations
                disc_X_grads = tape.gradient(disc_X_loss, self.X.trainable_variables)
                self.X_optimizer.apply_gradients(zip(disc_X_grads, self.X.trainable_variables))
                           
            test_x,test_y=next(iter(test_ds))            
            test_results = self.test_step(real_x=test_x, fake_x=test_y)
        
            trainresult.append(disc_X_loss.numpy())
            testresult.append(test_results)
            print('Epoch: %s'%epoch)
        return trainresult,testresult
        
#%%
mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
outputpath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/output'
weightoutputpath='/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/30102021/alpha'
# imgshape=(512,512)

cGAN=CycleGAN(mypath,weightoutputpath,NumberofTrainingSamples=1000,epochs=10,batch_size=10,imgshape=(512,512,1),patch_size=16,depth_size=32,totalbatchiterations=1,saveweightflag=True)
# cGAN.train_network()
lrs=cGAN.learningrate_log_scheduler()
train_res,test_res=cGAN.learning_rate_identifier()
# train_res,test_res=cGAN.learning_rate_identifier_batch()

mdic={"lrs":lrs,"train_res":train_res, "test_res":test_res}
savemat("LearningRateScheduler_Result.mat", mdic)
#%%
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)