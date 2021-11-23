#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:23:33 2021

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
        CT=(CT-np.min(CT))/np.ptp(CT)
        CTsiz1=CT.shape  
        CBCTCellRef=mat_contents['CBCTInfocell']
        CBCLen=np.shape(CBCTCellRef)
        CBCTi=np.random.choice(CBCLen[1],replace=False)  
        CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
        CBCellRef=mat_contents[CBCellRef]
        CBCT=CBCellRef[()]
        CBCT=np.transpose(CBCT,(2,1,0))
        CBCT=(CBCT-np.min(CBCT))/np.ptp(CBCT)
        CBsiz=CBCT.shape
        i=np.random.randint(CTsiz1[0]-self.patch_size*2)
        j=np.random.randint(CTsiz1[1]-self.patch_size*2)
        zi1=np.random.randint(CTsiz1[2]-self.depth_size)
        zi2=np.random.randint(CBsiz[2]-self.depth_size)
        CTblocks=CT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi1:zi1+self.depth_size]
        CBblocks=CBCT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi2:zi2+self.depth_size]
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
        CT=(CT-np.min(CT))/np.ptp(CT)
        CTsiz1=CT.shape  
        CBCTCellRef=mat_contents['CBCTInfocell']
        CBCLen=np.shape(CBCTCellRef)
        CBCTi=np.random.choice(CBCLen[1],replace=False)  
        CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
        CBCellRef=mat_contents[CBCellRef]
        CBCT=CBCellRef[()]
        CBCT=np.transpose(CBCT,(2,1,0))
        CBCT=(CBCT-np.min(CBCT))/np.ptp(CBCT)
        CBsiz=CBCT.shape
        i=np.random.randint(CTsiz1[0]-self.patch_size*2)
        j=np.random.randint(CTsiz1[1]-self.patch_size*2)
        zi1=np.random.randint(CTsiz1[2]-self.depth_size)
        zi2=np.random.randint(CBsiz[2]-self.depth_size)
        CTblocks=CT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi1:zi1+self.depth_size]
        CBblocks=CBCT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi2:zi2+self.depth_size]
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
         
         self.current_learning_rate = 0.0002
         self.learning_rate = 0.0002
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
         with open('Gena.txt', 'w+') as f:
             self.G.summary(print_fn=lambda x: f.write(x + '\n'))
         # self.GenCB2CT3D.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        
         self.X=self.build_discriminator3D(self.input_layer_shape_3D)
         # # self.DiscCT3D.summary()
         with open('Disc.txt', 'w+') as f:
             self.X.summary(print_fn=lambda x: f.write(x + '\n'))
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
         # self.mycompile()
         
         # with open('cycleGAN.txt', 'w+') as f:
         #     self.summary(print_fn=lambda x: f.write(x + '\n'))
         os.chdir(self.WeightSavePath)
         self.mycompile()
         self.checkpoint = tf.train.Checkpoint(G=self.G,
                                             F=self.F,
                                             X=self.X,
                                             Y=self.Y,
                                             G_optimizer=self.G_optimizer,
                                             F_optimizer=self.F_optimizer,
                                             X_optimizer=self.X_optimizer,
                                             Y_optimizer=self.Y_optimizer)
         
         self.checkpoint_dir = os.path.join(self.WeightSavePath, 'checkpoints')
         if not os.path.exists(self.checkpoint_dir):
             os.makedirs(self.checkpoint_dir)
        
         

    def save_checkpoint(self,chkfname):
        """ save checkpoint to checkpoint_dir, overwrite if exists """
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, chkfname)
        self.checkpoint.write(self.checkpoint_prefix)
        print(f'\nsaved checkpoint to {self.checkpoint_prefix}\n')

    def load_checkpoint(self, expect_partial: bool = False):
        """ load checkpoint from checkpoint_dir if exists """
        if os.path.exists(f'{os.path.join(self.checkpoint_prefix)}.index'):
            if expect_partial:
                self.checkpoint.read(self.checkpoint_prefix).expect_partial()
            else:
                self.checkpoint.read(self.checkpoint_prefix)
                print(f'\nloaded checkpoint from {self.checkpoint_prefix}\n')
    
    

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
        self.G_optimizer = keras.optimizers.Adam(learning_rate=self.current_learning_rate, beta_1=0.5)
        self.F_optimizer = keras.optimizers.Adam(learning_rate=self.current_learning_rate, beta_1=0.5)
        self.X_optimizer = keras.optimizers.Adam(learning_rate=self.current_learning_rate, beta_1=0.5)
        self.Y_optimizer = keras.optimizers.Adam(learning_rate=self.current_learning_rate, beta_1=0.5)
        self.generator_loss_fn = self.generator_loss_fnx
        self.discriminator_loss_fn = self.discriminator_loss_fnx
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
    
    def mycompile_1(self):
        super(CycleGAN,self).compile()
        self.G_optimizer = keras.optimizers.Adam(self.current_learning_rate, beta_1=0.5)
        self.F_optimizer = keras.optimizers.Adam(self.current_learning_rate, beta_1=0.5)
        self.X_optimizer = keras.optimizers.Adam(self.current_learning_rate, beta_1=0.5)
        self.Y_optimizer = keras.optimizers.Adam(self.current_learning_rate, beta_1=0.5)
        self.generator_loss_fn = self.generator_loss_fnx
        self.discriminator_loss_fn = self.discriminator_loss_fnx
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
    # @tf.function
    def train_step(self, real_x, real_y):
        # self.mycompile_1()
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
            
            # return {"G_loss": total_loss_G,"F_loss": total_loss_F,"D_X_loss": disc_X_loss,"D_Y_loss": disc_Y_loss}
            # return [total_loss_G, total_loss_F,disc_X_loss, disc_Y_loss]
            return [total_loss_G.numpy(), total_loss_F.numpy(),disc_X_loss.numpy(), disc_Y_loss.numpy()]
    # @tf.function    
    def test_step(self, real_x, real_y):
        self.mycompile_1()
# real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:
     
 # Horse to fake zebra
            fake_y = self.G(real_x, training=False)
            # Zebra to fake horse -> y2x
            fake_x = self.F(real_y, training=False)
            
            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.F(fake_y, training=False)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.G(fake_x, training=False)
            
            # Identity mapping
            same_x = self.F(real_x, training=False)
            same_y = self.G(real_y, training=False)
            
            # Discriminator output
            disc_real_x = self.X(real_x, training=False)
            disc_fake_x = self.X(fake_x, training=False)
            
            disc_real_y = self.Y(real_y, training=False)
            disc_fake_y = self.Y(fake_y, training=False)
            
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
            # grads_G = tape.gradient(total_loss_G, self.G.trainable_variables)
            # grads_F = tape.gradient(total_loss_F, self.F.trainable_variables)
            
            # # Get the gradients for the discriminators
            # disc_X_grads = tape.gradient(disc_X_loss, self.X.trainable_variables)
            # disc_Y_grads = tape.gradient(disc_Y_loss, self.Y.trainable_variables)
            
            # # Update the weights of the generators
            # self.G_optimizer.apply_gradients(zip(grads_G, self.G.trainable_variables))
            # self.F_optimizer.apply_gradients(zip(grads_F, self.F.trainable_variables))
            
            # # Update the weights of the discriminators
            # self.X_optimizer.apply_gradients(zip(disc_X_grads, self.X.trainable_variables))
            # self.Y_optimizer.apply_gradients(zip(disc_Y_grads, self.Y.trainable_variables))
            
            # return {"G_loss": total_loss_G,"F_loss": total_loss_F,"D_X_loss": disc_X_loss,"D_Y_loss": disc_Y_loss}
            # return [total_loss_G, total_loss_F,disc_X_loss, disc_Y_loss]
            return [total_loss_G.numpy(), total_loss_F.numpy(),disc_X_loss.numpy(), disc_Y_loss.numpy()]
        
    #Learning rate scheduler to jump local minima or plateau
    def cyclic_learning_rate_scheduler(self):
        lr_max=0.003
        lr=0.0002
        fullcyc=int(self.epochs/8)
        # epochs=200
        cycles_N=int(self.epochs/fullcyc)
        epochtime=np.linspace(0, fullcyc,fullcyc)
        # epochfulltime=np.linspace(0, self.epochs,self.epochs)
        sig=np.zeros((self.epochs,))
        for cycle in range(cycles_N):
            lr_pp=lr_max-lr
            sig3=(sg.sawtooth(np.pi*2*epochtime,width=0.5)+1)*lr_pp
            lr_max=lr_max/2
            sig[cycle*(fullcyc):cycle*(fullcyc)+fullcyc]=sig3
        sig[0]=sig[1]
        return sig
    
    def batch_iteration_scheduler(self):
        totalbatchiterations = self.totalbatchiterations/2
        batch_iterations=np.zeros((self.epochs,))
        for epoch in range(self.epochs):
            if epoch % 50 == 0:
                totalbatchiterations = totalbatchiterations*2
            batch_iterations[epoch]=totalbatchiterations
        return batch_iterations
                
        
    
    def train_network_full(self):

        train_ds = self.get_ds()
        test_ds = self.get_ds_test()
        learning_rates=self.cyclic_learning_rate_scheduler()
        batch_iterations=self.batch_iteration_scheduler()
        trainresult=[]
        testresult=[]
        
        for epoch in range(self.epochs):
            # Adaptive learning rate
            if epoch <100:
                self.current_learning_rate=learning_rates[epoch]
            else:
                self.current_learning_rate=self.learning_rate
            
            self.totalbatchiterations=int(batch_iterations[epoch])
                
            # Adaptive batch size or batch iterations
            for batch_index in range(self.totalbatchiterations):
                x,y=next(iter(train_ds))
                test_x,test_y=next(iter(test_ds))
                train_results = self.train_step(real_x=x, real_y=y)
                test_results = self.test_step(real_x=test_x, real_y=test_y)
                # print(train_results)
                # print('Current batch: %s'%batch_index)
            if epoch % 50 == 0 and self.saveweightflag==True:
                self.save_checkpoint(str(epoch))
            trainresult.append(train_results)
            testresult.append(test_results)
            
        return trainresult,testresult
            
                
    
    def train_network(self):


        train_ds = self.get_ds()
        test_ds = self.get_ds_test()
        # learning_rates=self.cyclic_learning_rate_scheduler()
        # batch_iterations=self.batch_iteration_scheduler()
        trainresult=[]
        testresult=[]
        for epoch in range(self.epochs):
            self.current_learning_rate=self.learning_rate

            for batch_index in range(self.totalbatchiterations):
                x,y=next(iter(train_ds))
                test_x,test_y=next(iter(test_ds))
                train_results = self.train_step(real_x=x, real_y=y)
                test_results = self.test_step(real_x=test_x, real_y=test_y)
        
            trainresult.append(train_results)
            testresult.append(test_results)
            # self.save_checkpoint(str(epoch))
            print('Epoch: %s'%epoch)
        
            
        return trainresult,testresult
            
            # if epoch % 1 == 0 and self.saveweightflag==True: # Weights saved based on epoch intervals and checkpoints
            #     self.save_checkpoint()
            #     # gen1fname1=gen1fname+'-'+str(epoch)+'.h5'    
            #     # gen2fname1=gen2fname+'-'+str(epoch)+'.h5'
            #     # disc1fname1=disc1fname+'-'+str(epoch)+'.h5'
            #     # disc2fname1=disc2fname+'-'+str(epoch)+'.h5'
            #     # self.G.save_weights(gen1fname1)
            #     # self.F.save_weights(gen2fname1)
            #     # self.X.save_weights(disc1fname1)
            #     # self.Y.save_weights(disc2fname1)
            #     print ('r')
                
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


cGAN=CycleGAN(mypath,weightoutputpath,NumberofTrainingSamples=1000,epochs=100,batch_size=1,imgshape=(512,512,1),patch_size=16,depth_size=32,totalbatchiterations=10,saveweightflag=True)
# cGAN.load_checkpoint()
train_res,test_res=cGAN.train_network()
# savemat('trainloss.mat', train_res)
# savemat('testloss.mat', test_res)
# cGAN.load_checkpoint()
# cGAN.dataload3D_2_test()
#saving the variables
import pickle
with open("train_losses.pkl", "wb") as f:
    pickle.dump(train_res, f)
with open("test_losses.pkl", "wb") as f:
    pickle.dump(test_res, f)
#loading it for plotting
# with open("train_losses.pkl","rb") as f1:
#     train_res_loaded = pickle.load(f1)
mdic={"train_res":train_res, "test_res":test_res}
savemat("TrainResult.mat", mdic)

#%%


# with open("cGAN.pkl", "wb") as f:
#     pickle.dump(cGAN, f)

# lrs=cGAN.cyclic_learning_rate_scheduler()
# bs=cGAN.batch_iteration_scheduler()
    
# cGAN.save_checkpoint(100)

#%%
test=cGAN.dataload3D_2_test()
testx,testy=next(iter(test))
test_xp=cGAN.G(testx,training=False)

xt=testx.numpy()
xp=test_xp.numpy()
xt=np.squeeze(xt,axis=0)
xp=np.squeeze(xp,axis=0)
xp=np.squeeze(xp,axis=-1)

import matplotlib.pyplot as plt
plt.figure(1),plt.subplot(1,2,1),plt.imshow(xt[:,:,24],cmap='gray'),plt.show()
plt.figure(1),plt.subplot(1,2,2),plt.imshow(xp[:,:,24],cmap='gray'),plt.show()

#%%  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)