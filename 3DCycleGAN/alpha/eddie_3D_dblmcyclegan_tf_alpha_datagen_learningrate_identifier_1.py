#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:38:26 2021

@author: arun
"""
import time
import datetime
# import cv2
# import itk
# import h5py
import numpy as np
# from os import listdir
# from os.path import isfile, join
# import matplotlib.pyplot as plt
import random
import os
import sys, getopt
# import multiprocessing


# os.environ['CUDA_VISIBLE_DEVICES'] = "-1" # comment this line when running in eddie
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import scipy.io
from tensorflow.keras.utils import Sequence


tf.keras.backend.clear_session()

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
#%% Data loader using data generator

def create_image_array_gen_CT(trainCT_image_names, trainCT_path):
    image_array = []
    for image_name in trainCT_image_names:
        mat_contents=scipy.io.loadmat(os.path.join(trainCT_path,image_name))
        CT_b=mat_contents['CT_b']
        CT_b=np.array(CT_b)
        # CT_b = ((CT_b-np.min(CT_b))/((np.max(CT_b)-np.min(CT_b))*0.5))-1#Normalisation needs proper
        CT_b=np.expand_dims(CT_b, axis=-1)
        image_array.append(CT_b)
    return np.array(image_array)

def create_image_array_gen_CB(trainCT_image_names, trainCT_path):
    image_array = []
    for image_name in trainCT_image_names:
        mat_contents=scipy.io.loadmat(os.path.join(trainCT_path,image_name))
        CT_b=mat_contents['CB_b']
        CT_b=np.array(CT_b)
        # CT_b = 2.*(CT_b-np.min(CT_b))/(np.max(CT_b)-np.min(CT_b))-1
        # CT_b = ((CT_b-np.min(CT_b))/((np.max(CT_b)-np.min(CT_b))*0.5))-1
        CT_b=np.expand_dims(CT_b, axis=-1)
        image_array.append(CT_b)
    return np.array(image_array)

class data_sequence(Sequence):
    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size,batch_set_size):
        # self.newshape=newshape
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        self.batch_set_size=batch_set_size
        image_list_A=random.sample(image_list_A,self.batch_set_size)
        image_list_B=random.sample(image_list_B,self.batch_set_size)
        for image_name in image_list_A:
            # if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            # if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))
    
    def __len__(self):
        no=1
        # return int(min(len(self.train_A), len(self.train_B)) / float(self.batch_size))
        return int(no)
    
    def __getitem__(self, idx):
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
        real_images_A = create_image_array_gen_CT(batch_A, '')
        real_images_B = create_image_array_gen_CB(batch_B, '')
        return real_images_A, real_images_B  # input_data, target_data
                

def loadprintoutgen(trainCT_path,trainCB_path,batch_size,batch_set_size):
    trainCT_image_names = os.listdir(trainCT_path)
    trainCB_image_names = os.listdir(trainCB_path)
    # return trainCT_image_names,trainCB_image_names
    return data_sequence(trainCT_path, trainCB_path, trainCT_image_names, trainCB_image_names,batch_size=batch_size,batch_set_size=batch_set_size)

#%%

class CycleGAN():
    
#%% 2D     
    @staticmethod
    def conv2d(layer_input, filters, f_size=4,stride=2,normalization=True):
        """Discriminator layer"""
        d = layers.Conv2D(filters, kernel_size=f_size,strides=stride, padding='same',activation='relu')(layer_input)
        if normalization:
            d = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(d)
        d = layers.LeakyReLU(alpha=0.2)(d)
        return d
    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, stride=1, dropout_rate=0,skip=True):
          """Layers used during upsampling"""
          u = layers.UpSampling2D(size=2)(layer_input)
          u = layers.Conv2D(filters, kernel_size=f_size, strides=stride, padding='same', activation='tanh')(u)
          if dropout_rate:
              u = layers.Dropout(dropout_rate)(u)
          u = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(u)
          if skip:
              u = layers.Concatenate()([u, skip_input])
          return u
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
    
    def build_generator3D(self):
        ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
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
    
    def build_discriminator3D(self):
        ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
        opL1=self.convblk3d(ipL,self.discfilter,self.kernel_size_disc,self.stride2)
        opL2=self.convblk3d(opL1,self.discfilter*2,self.kernel_size_disc,self.stride2)
        opL3=self.convblk3d(opL2,self.discfilter*4,self.kernel_size_disc,self.stride2)
        opL4=self.convblk3d(opL3,self.discfilter*8,self.kernel_size_disc,self.stride1)
        
        # opL5=layers.Flatten()(opL4)
        
        # opL5=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL5)
        # opL5=layers.LeakyReLU()(opL5)
        
        # opL6=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*16)(opL5)
        # opL6=layers.LeakyReLU()(opL6)

        # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL6)
        # opL8=layers.Activation('sigmoid')(opL7)

        # opL9=layers.Dense(1)(opL8)
        # opL=layers.Activation('sigmoid')(opL9)
        
        
        opL5=layers.Conv3D(1, kernel_size=self.kernel_size_disc, strides=self.stride1,padding='SAME')(opL4)
        
        opL7 = layers.Flatten()(opL5)
        
        # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL5)
        # opL7=layers.LeakyReLU()(opL7)
        
        # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*16)(opL7)
        # opL7=layers.LeakyReLU()(opL7)
        
        # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*8))(opL7)
        # opL7=layers.LeakyReLU()(opL7)
        
        # opL7=layers.Conv3D(1, kernel_size=self.kernel_size_disc, strides=self.stride1,padding='SAME')(opL6)
        
        opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter))(opL7)
        opL7=layers.LeakyReLU()(opL7)
        
        opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.5))(opL7)
        opL7=layers.LeakyReLU()(opL7)
        
        opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.25))(opL7)
        opL7=layers.LeakyReLU()(opL7)
        
        opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.125))(opL7)
        opL7=layers.LeakyReLU()(opL7)
        
        opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.0625))(opL7)
        opL7=layers.LeakyReLU()(opL7)
        
        opL7 = layers.Dense(1)(opL7)
        opL = layers.Activation('relu')(opL7)
        
        return keras.Model(ipL,opL)
    
    def build_discriminator(self):
        d0 = keras.Input(shape=self.img_shape,name='Input')
        d1 = self.conv2d(d0, self.discfilter, stride=2, normalization=False)
        d2 = self.conv2d(d1, self.discfilter*2, stride=2, normalization=True)
        d3 = self.conv2d(d2, self.discfilter*4, stride=2, normalization=True)
        d4 = self.conv2d(d3, self.discfilter*8, stride=2, normalization=True)
        d5 = self.conv2d(d4, self.discfilter*8, stride=1, normalization=True)
        d6 = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)
        d7 = layers.Activation('relu')(d6)
        
        return keras.Model(d0,d7)
    
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
        u1 = layers.Activation('tanh')(u1)
        
        return keras.Model(d0,u1)
    
    def __init__(self,mypath,weightoutputpath,epochs,batch_size,imgshape,batch_set_size,saveweightflag):
          self.DataPath=mypath
          self.WeightSavePath=weightoutputpath
          self.batch_size=batch_size
          self.img_shape=imgshape
          self.input_shape=tuple([batch_size,imgshape])
          self.genafilter = 16
          self.discfilter = 16
          self.epochs = epochs+1
          self.batch_set_size=batch_set_size
          self.lambda_cycle = 10.0                    # Cycle-consistency loss
          self.lambda_id = 0.9 * self.lambda_cycle    # Identity loss
          self.saveweightflag=saveweightflag
          self.patch_size=32
          self.depth_size=32
          self.input_layer_shape_3D=tuple([self.patch_size,self.patch_size,self.depth_size,1])
          self.stride2=2
          self.stride1=1
          self.kernel_size = 3
          self.kernel_size_disc = 4
         
          os.chdir(self.WeightSavePath)
          self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))
          os.mkdir(self.folderlen)
          self.WeightSavePath = os.path.join(self.WeightSavePath, self.folderlen)
          os.chdir(self.WeightSavePath)
         
          newdir='arch'
         
          self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
          os.mkdir(self.WeightSavePathNew)
          os.chdir(self.WeightSavePathNew)
          
          self.Disc_lr=0.0002
          self.Gen_lr=0.0002
         
          self.Disc_optimizer = keras.optimizers.Adam(self.Disc_lr, 0.5,0.999)
          self.Gen_optimizer = keras.optimizers.Adam(self.Gen_lr, 0.5,0.999)
         
          # optimizer = keras.optimizers.Adam(0.0002, 0.5)

          self.DiscCT=self.build_discriminator3D()
          self.DiscCT.compile(loss='mse', optimizer=self.Disc_optimizer, metrics=['accuracy'])
          self.DiscCT._name='Discriminator-CT'
          # self.DiscCT.summary()
          with open('Disc.txt', 'w+') as f:
              self.DiscCT.summary(print_fn=lambda x: f.write(x + '\n'))
         
          self.DiscCB=self.build_discriminator3D()
          self.DiscCB.compile(loss='mse', optimizer=self.Disc_optimizer, metrics=['accuracy'])
          self.DiscCB._name='Discriminator-CB'
         
          layer_len=len(self.DiscCT.layers)
          layers_lis=self.DiscCT.layers
          labelshapearr=list(layers_lis[layer_len-1].output_shape)
          labelshapearr[0]=self.batch_size
          labelshape=tuple(labelshapearr)
          self.labelshape=labelshape
        
          self.GenCB2CT=self.build_generator3D()
          self.GenCB2CT.compile(loss='mse', optimizer=self.Gen_optimizer, metrics=['accuracy'])
          self.GenCB2CT._name='Generator-CB2CT'
          # self.GenCB2CT.summary()
          with open('Gena.txt', 'w+') as f:
              self.GenCB2CT.summary(print_fn=lambda x: f.write(x + '\n'))
         
          self.GenCT2CB=self.build_generator3D()
          self.GenCT2CB.compile(loss='mse', optimizer=self.Gen_optimizer, metrics=['accuracy'])
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
          loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id], optimizer=self.Gen_optimizer)
          self.cycleGAN_Model._name='CycleGAN'
         
          with open('cycleGAN.txt', 'w+') as f:
              self.cycleGAN_Model.summary(print_fn=lambda x: f.write(x + '\n'))
          
          trainCT_path = os.path.join(self.DataPath, 'trainCT')
          trainCB_path = os.path.join(self.DataPath, 'trainCB')
          # testCT_path = os.path.join(self.DataPath, 'validCT')
          # testCB_path = os.path.join(self.DataPath, 'validCB')
          self.data_generator=loadprintoutgen(trainCT_path,trainCB_path,self.batch_size,self.batch_set_size)
          
          # for images in self.data_generator:
          #     batch_CT = images[0]
          #     batch_CB = images[1]
              
          #     print(batch_CT.shape)
          print(self.data_generator.__len__())
          os.system("nvidia-smi")
          print('Init')
          
    def learningrate_log_scheduler(self):
        learning_rates=np.logspace(-8, 1,num=self.epochs)
        return learning_rates
    
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
        # D_losses = np.zeros((self.batch_set_size,2,self.epochs))
        # G_losses = np.zeros((self.batch_set_size,7,self.epochs))
        D_losses = []
        G_losses = []
        
        # D_loss_dict = []
        # D_loss_dict.append([])
        # D_loss_dict.append([])
        # G_loss_dict = []
        # G_loss_dict.append([])
        # G_loss_dict.append([])
        D_loss_dict = np.zeros((self.epochs,self.epochs))
        G_loss_dict = np.zeros((self.epochs,self.epochs))
        #Learning rate schedule
        learning_rates=self.learningrate_log_scheduler()
        
        def run_training_iteration(loop_index, epoch_iterations):
              valid = tf.ones((self.labelshape))
              fake = tf.zeros((self.labelshape))
                 # ----------------------
                 #  Train Discriminators
                 # ----------------------

                 # Translate images to opposite domain
              fake_CB = self.GenCT2CB.predict(batch_CT)
              fake_CT = self.GenCB2CT.predict(batch_CB)

                 # Train the discriminators (original images = real / translated = Fake)
              dCT_loss_real = self.DiscCT.train_on_batch(batch_CT, valid)
              dCT_loss_fake = self.DiscCT.train_on_batch(fake_CT, fake)
              dCT_loss = 0.5 * tf.math.add(dCT_loss_real, dCT_loss_fake)

              dCB_loss_real = self.DiscCB.train_on_batch(batch_CB, valid)
              dCB_loss_fake = self.DiscCB.train_on_batch(fake_CB, fake)
              dCB_loss = 0.5 * tf.math.add(dCB_loss_real, dCB_loss_fake)

                 # Total discriminator loss
              d_loss = 0.5 * tf.math.add(dCT_loss, dCB_loss)

                 # ------------------
                 #  Train Generators
                 # ------------------

                 # Train the generators
              g_loss = self.cycleGAN_Model.train_on_batch([batch_CT, batch_CB],
                                                       [valid, valid,
                                                        batch_CT, batch_CB,
                                                        batch_CT, batch_CB])
              
              return g_loss, d_loss.numpy()
              
        for epochi in range(self.epochs):
            # if self.use_data_generator:
            loop_index = 1
                
            K.set_value(self.Gen_optimizer.learning_rate, learning_rates[epochi])
            for epochi2 in range(self.epochs):
                K.set_value(self.Disc_optimizer.learning_rate, learning_rates[epochi2])
                
                for images in self.data_generator:
                    batch_CT = images[0]
                    batch_CB = images[1]
                    # Run all training steps
                    g_loss, d_loss=run_training_iteration(loop_index, self.data_generator.__len__())
                    os.system("nvidia-smi")
                    #print('Trainingstep')
                    
                    if loop_index >= self.data_generator.__len__():
                        break
                    loop_index = loop_index+1
                    
                    
                
                D_losses.append(d_loss)
                G_losses.append(g_loss)
                D_loss_dict[epochi,epochi2]=d_loss[0]
                G_loss_dict[epochi,epochi2]=g_loss[0]
                # D_loss_dict.append(D_losses)
                # G_loss_dict.append(G_losses)
                    
            if epochi % 1 == 0 and self.saveweightflag==True: # Weights saved based on epoch intervals
                gen1fname1=gen1fname+'-'+str(epochi)+'.h5'    
                gen2fname1=gen2fname+'-'+str(epochi)+'.h5'
                disc1fname1=disc1fname+'-'+str(epochi)+'.h5'
                disc2fname1=disc2fname+'-'+str(epochi)+'.h5'
                self.GenCT2CB.save_weights(gen1fname1)
                self.GenCB2CT.save_weights(gen2fname1)
                self.DiscCT.save_weights(disc1fname1)
                self.DiscCB.save_weights(disc2fname1)
        os.system("nvidia-smi")
        print('Epoch')
        return D_losses,G_losses,D_loss_dict,G_loss_dict
        
#%%

mypath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/'
outputpath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/output'
weightoutputpath='/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/01112021/alpha/'
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
cGAN=CycleGAN(mypath,weightoutputpath,epochs=20,batch_size=5,imgshape=(256,256,1),batch_set_size=100,saveweightflag=False)
# def run_tf(cGAN):
#     D_losses,G_losses=cGAN.traincgan()
    
# p=multiprocessing.Process(target=run_tf(cGAN))
# p.start()
# p.join()

# D_losses,G_losses=cGAN.traincgan()
# data=cGAN.data_generator()

D_losses,G_losses,D_loss_dict,G_loss_dict=cGAN.traincgan()
lr=cGAN.learningrate_log_scheduler()
#%%
from scipy.io import savemat
mdic = {"D_losses":D_loss_dict,"G_losses":G_loss_dict,"lr":lr}
savemat("Losses.mat",mdic)

# train_ds=cGAN.get_ds()

# batch_CT, batch_CB, G_losses, D_losses=cGAN.traincgan()

# batch_CT, batch_CB =cGAN.dataload()
# GenCT2CBWeightsFileName="/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/weights/GenCT2CBWeights-200.h5"
# batch_CB_P=cGAN.predictCT2CB(batch_CT,batch_CB,GenCT2CBWeightsFileName)

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
tf.keras.backend.clear_session()