#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:51:26 2021

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

import ssimetricTFlib as ssTF

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

# batch_CT=batch_CT1[:,:,2]
# batch_CB=batch_CB1[:,:,2]

batch_CT = np.transpose(batch_CT,(2,0,1))
batch_CT = np.expand_dims(batch_CT, -1)
batch_CB = np.transpose(batch_CB,(2,0,1))
batch_CB = np.expand_dims(batch_CB, -1)

imgshape=(512,512,1)
#%%

def gaussian_blur(img, kernel_size=11, sigma=1.5):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.conv2d(img, gaussian_kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC')

# Not an ideal approach to design a mean filter. The gaussian kernel was resetted to ones and divided by it sum.
# this was done to save time in coding not computationally effective
# The syntax for writing a new sub-routine took a lot of time
def uniform_filter_tf(img, kernel_size=11, sigma=1.5):
    def uniform_kernel_fx(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = tf.ones_like(kernel)
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    u_kernel = uniform_kernel_fx(tf.shape(img)[-1], kernel_size, sigma)
    u_kernel = u_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, u_kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC')

def mean_filter2d(image, filter_shape, name=None):
    area = filter_shape[0] * filter_shape[1]
    filter_shape = filter_shape + (tf.shape(image)[-1], 1)
    kernel = tf.ones(shape=filter_shape) / area
    # output = tf.nn.depthwise_conv2d(image, kernel, strides=(1, 1, 1, 1), padding="SAME")
    return tf.nn.conv2d(image, kernel, strides=(1, 1, 1, 1), padding="SAME")

def myssimtf(di1,di2,imgshape):
    # di1=di1.astype(int)
    di1=tf.convert_to_tensor(di1)
    # tf.dtypes.cast(di1, tf.int16)
    # di1 = keras.Input(shape=imgshape,name='Input1')#Input
    dy1, dx1 = tf.image.image_gradients(di1)#Gradient directional maps
    d_gm1 = 0.5 *(dy1+dx1)# Gradient magnitude
    d_r1 = (tf.math.reduce_max(d_gm1)-tf.math.reduce_min(d_gm1))#Dynamic Range for SSIM calculation
    d_mu1 = gaussian_blur(di1, kernel_size=11, sigma=1.5)
    
    di2=tf.convert_to_tensor(di2)
    # tf.dtypes.cast(di2, tf.int16)
    
    # di2=di2.astype(int)
    # di2 = keras.Input(shape=imgshape,name='Input2')#Input
    dy2, dx2 = tf.image.image_gradients(di1)#Gradient directional maps
    d_gm2 = 0.5 *(dy2+dx2)# Gradient magnitude
    d_r2 = (tf.math.reduce_max(d_gm2)-tf.math.reduce_min(d_gm2))#Dynamic Range for SSIM calculation
    d_mu2 = gaussian_blur(di2, kernel_size=11, sigma=1.5)
    
    # d2 = uniform_filter_tf(d0, kernel_size=11)
    d_sigma1_sq = uniform_filter_tf(d_gm1, kernel_size=11)
    d_sigma2_sq = uniform_filter_tf(d_gm2, kernel_size=11)
    # d_m3 = uniform_filter_tf(tf.math.sqrt(d_m*d_m), kernel_size=11)
    d_sigma12 = uniform_filter_tf(tf.math.sqrt(d_gm1*d_gm2), kernel_size=11)
    
    d_r=0.5*(d_r1+d_r2)
    C1 = (0.01 * d_r)**2
    C2 = (0.03 * d_r)**2
    
    d_mu1_sq = d_mu1**2
    d_mu2_sq = d_mu2**2
    d_mu1_mu2 = d_mu1 * d_mu2
    
    ssim_map = ((2 * d_mu1_mu2 + C1) * (2 * d_sigma12 + C2)) / ((d_mu1_sq + d_mu2_sq + C1) * (d_sigma1_sq + d_sigma2_sq + C2))
    ssim_score =tf.math.reduce_mean(ssim_map)
    # return keras.Model(inputs=[di1,di2], outputs=[ssim_map,ssim_score])
    return ssim_map,ssim_score

def myssimtf_1(imgshape):
    # di1=di1.astype(int)
    # di1=tf.convert_to_tensor(di1)
    # tf.dtypes.cast(di1, tf.int16)
    di1 = keras.Input(shape=imgshape,name='Input1')#Input
    dy1, dx1 = tf.image.image_gradients(di1)#Gradient directional maps
    d_gm1 = 0.5 *(dy1+dx1)# Gradient magnitude
    d_r1 = (tf.math.reduce_max(d_gm1)-tf.math.reduce_min(d_gm1))#Dynamic Range for SSIM calculation
    d_mu1 = gaussian_blur(di1, kernel_size=11, sigma=1.5)
    
    # di2=tf.convert_to_tensor(di2)
    # tf.dtypes.cast(di2, tf.int16)
    # di2=di2.astype(int)
    di2 = keras.Input(shape=imgshape,name='Input2')#Input
    dy2, dx2 = tf.image.image_gradients(di1)#Gradient directional maps
    d_gm2 = 0.5 *(dy2+dx2)# Gradient magnitude
    d_r2 = (tf.math.reduce_max(d_gm2)-tf.math.reduce_min(d_gm2))#Dynamic Range for SSIM calculation
    d_mu2 = gaussian_blur(di2, kernel_size=11, sigma=1.5)
    
    # d2 = uniform_filter_tf(d0, kernel_size=11)
    d_sigma1_sq = uniform_filter_tf(d_gm1, kernel_size=11)
    d_sigma2_sq = uniform_filter_tf(d_gm2, kernel_size=11)
    # d_m3 = uniform_filter_tf(tf.math.sqrt(d_m*d_m), kernel_size=11)
    d_sigma12 = uniform_filter_tf(tf.math.sqrt(d_gm1*d_gm2), kernel_size=11)
    
    d_r=0.5*(d_r1+d_r2)
    C1 = (0.01 * d_r)**2
    C2 = (0.03 * d_r)**2
    
    d_mu1_sq = d_mu1**2
    d_mu2_sq = d_mu2**2
    d_mu1_mu2 = d_mu1 * d_mu2
    
    ssim_map = ((2 * d_mu1_mu2 + C1) * (2 * d_sigma12 + C2)) / ((d_mu1_sq + d_mu2_sq + C1) * (d_sigma1_sq + d_sigma2_sq + C2))
    ssim_score =tf.math.reduce_mean(ssim_map)
    return keras.Model(inputs=[di1,di2], outputs=[ssim_map,ssim_score])
    # return ssim_map,ssim_score


# ssim_map,ssim_score=myssimtf(batch_CT,batch_CT,imgshape)
SSIMTF=myssimtf_1(imgshape)

o1,o2=SSIMTF.predict([batch_CT,batch_CB])

o1n = np.squeeze(o1, axis=3)
o1n = np.transpose(o1n,(1,2,0))
batch_CTn = np.squeeze(batch_CT, axis=3)
batch_CTn = np.transpose(batch_CTn,(1,2,0))
batch_CBn = np.squeeze(batch_CB, axis=3)
batch_CBn = np.transpose(batch_CBn,(1,2,0))
#%%
plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(o1n, cmap='gray'),plt.show()
plt.subplot(1,3,2)
plt.imshow(batch_CTn, cmap='gray'),plt.show()
plt.subplot(1,3,3)
plt.imshow(batch_CBn, cmap='gray'),plt.show()

#%%
# os.chdir(scriptpath)

# import ssimmetricslib

# import ssimetricTFlib as ssTF
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops

def _verify_compatible_image_shapes1(img1, img2):
  shape1 = img1.get_shape().with_rank_at_least(3)
  shape2 = img2.get_shape().with_rank_at_least(3)
  shape1[-3:].assert_is_compatible_with(shape2[-3:])

  if shape1.ndims is not None and shape2.ndims is not None:
    for dim1, dim2 in zip(
        reversed(shape1.dims[:-3]), reversed(shape2.dims[:-3])):
      if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
        raise ValueError('Two images are not compatible: %s and %s' %
                         (shape1, shape2))

  # Now assign shape tensors.
  shape1, shape2 = array_ops.shape_n([img1, img2])

  # TODO(sjhwang): Check if shape1[:-3] and shape2[:-3] are broadcastable.
  checks = []
  checks.append(
      control_flow_ops.Assert(
          math_ops.greater_equal(array_ops.size(shape1), 3), [shape1, shape2],
          summarize=10))
  checks.append(
      control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(shape1[-3:], shape2[-3:])),
          [shape1, shape2],
          summarize=10))
  return shape1, shape2, checks

def _fspecial_gauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  size = ops.convert_to_tensor(size, tf.int32)
  sigma = ops.convert_to_tensor(sigma)

  coords = math_ops.cast(math_ops.range(size), sigma.dtype)
  coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0

  g = math_ops.square(coords)
  g *= -0.5 / math_ops.square(sigma)

  g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
  g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
  g = nn_ops.softmax(g)
  return array_ops.reshape(g, shape=[size, size, 1, 1])

def _ssim_helper_2(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):

  c1 = (k1 * max_val)**2
  

  # SSIM luminance measure is
  # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
  mean0 = reducer(x)
  mean1 = reducer(y)
  num0 = mean0 * mean1 * 2.0
  den0 = math_ops.square(mean0) + math_ops.square(mean1)
  luminance = (num0 + c1) / (den0 + c1)

  # SSIM contrast-structure measure is
  #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
  # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
  #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
  #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).

  # num1 = reducer(x * y) * 2.0
  # den1 = reducer(math_ops.square(x) + math_ops.square(y))
  # c2 *= compensation
  # cs = (num1 - num0 + c2) / (den1 - den0 + c2)
  
  dy1, dx1 = tf.image.image_gradients(x)#Gradient directional maps
  d_gm1 = 0.5 *(dy1+dx1)
  dy2, dx2 = tf.image.image_gradients(y)#Gradient directional maps
  d_gm2 = 0.5 *(dy2+dx2)
  
  c2 = (k2 * max_val)**2
  
  num1 = reducer(d_gm1 * d_gm2) * 2.0
  den1 = reducer(math_ops.square(d_gm1) + math_ops.square(d_gm2))
  c2 *= compensation
  cs = (num1 - num0 + c2) / (den1 - den0 + c2)
  
  smap_p=luminance * cs
  
  TH1=0.12*tf.math.reduce_max(d_gm1)
  TH2=0.06*tf.math.reduce_max(d_gm2)
  
  d_m1_mask_ones=tf.ones_like(d_gm1)# Create masks for 4-components and tf.where method
  d_m1_mask_zeros=tf.zeros_like(d_gm1)
  
  d_m1_mask_1=tf.math.logical_and(d_gm1>TH1,d_gm2>TH1)
  # d_m1_mask_1_ind=tf.where(tf.math.logical_and(d_gm1>TH1,d_gm2>TH1))# Indices for preserved edge mask
  d_m1_mask_r1=tf.where(d_m1_mask_1,d_m1_mask_zeros,d_m1_mask_ones)# Region mask for preserved edge mask - 1/4 components
  
  d_m1_mask_2=tf.math.logical_and(d_gm1>TH1,d_gm2<=TH1)
  # d_m1_mask_1_ind=tf.where(tf.math.logical_and(d_gm1>TH1,d_gm2>TH1))# Indices for changed edge mask
  d_m1_mask_r2=tf.where(d_m1_mask_2,d_m1_mask_zeros,d_m1_mask_ones)# Region mask for changed edge mask - 2/4 components
  
  d_m1_mask_3=tf.math.logical_and(d_gm1<TH2,d_gm2<TH2)
  # d_m1_mask_1_ind=tf.where(tf.math.logical_and(d_gm1>TH1,d_gm2>TH1))# Indices for smooth mask
  d_m1_mask_r3=tf.where(d_m1_mask_3,d_m1_mask_ones,d_m1_mask_zeros)# Region mask for smooth mask - 3/4 components
  
  d_mask_m12=tf.math.logical_or(d_m1_mask_1, d_m1_mask_2)
  d_mask_m123=tf.math.logical_or(d_mask_m12, d_m1_mask_3)# Union set operation by logical 'or' the regions R1, R2 and R3
  d_m1_mask_4=tf.math.logical_not(d_mask_m123)#Exclusion of regions R1, R2, and R3 by complementing the union set using logical 'not'
  d_m1_mask_r4=tf.where(d_m1_mask_4,d_m1_mask_zeros,d_m1_mask_ones)# Region mask for texture mask - 4/4 components
  
  smapR1=tf.math.multiply(smap_p, d_m1_mask_r1)
  smapR2=tf.math.multiply(smap_p, d_m1_mask_r2)
  smapR3=tf.math.multiply(smap_p, d_m1_mask_r3)
  smapR4=tf.math.multiply(smap_p, d_m1_mask_r4)
  
  smap=0.25*smapR1+0.25*smapR2+0.25*smapR3+0.25*smapR4

  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs, smap

def _ssim_per_channel_2(img1,
                      img2,
                      max_val=1.0,
                      filter_size=11,
                      filter_sigma=1.5,
                      k1=0.01,
                      k2=0.03):

  filter_size = constant_op.constant(filter_size, dtype=tf.int32)
  filter_sigma = constant_op.constant(filter_sigma, dtype=img1.dtype)

  shape1, shape2 = array_ops.shape_n([img1, img2])
  checks = [
      control_flow_ops.Assert(
          math_ops.reduce_all(
              math_ops.greater_equal(shape1[-3:-1], filter_size)),
          [shape1, filter_size],
          summarize=8),
      control_flow_ops.Assert(
          math_ops.reduce_all(
              math_ops.greater_equal(shape2[-3:-1], filter_size)),
          [shape2, filter_size],
          summarize=8)
  ]

  # Enforce the check to run before computation.
  with ops.control_dependencies(checks):
    img1 = array_ops.identity(img1)

  # TODO(sjhwang): Try to cache kernels and compensation factor.
  kernel = _fspecial_gauss(filter_size, filter_sigma)
  kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

  # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
  # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
  compensation = 1.0

  # TODO(sjhwang): Try FFT.
  # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
  #   1-by-n and n-by-1 Gaussian filters instead of an n-by-n filter.
  def reducer(x):
    shape = array_ops.shape(x)
    x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
    y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
    # Don't use padding=VALID
    # padding should be SAME to avoid resolution mismatch between smap and mask multiplication
    return array_ops.reshape(
        y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0))

  luminance, cs, smap = _ssim_helper_2(img1, img2, reducer, max_val, compensation, k1,
                               k2)

  # Average over the second and the third from the last: height, width.
  axes = constant_op.constant([-3, -2], dtype=tf.int32)
  # ssim_map = luminance * cs
  # ssim_val = math_ops.reduce_mean(luminance * cs, axes)
  ssim_val = math_ops.reduce_mean(smap, axes)
  cs = math_ops.reduce_mean(cs, axes)
  return ssim_val, cs


def ssim_2(img1,
         img2,
         max_val,
         filter_size=11,
         filter_sigma=1.5,
         k1=0.01,
         k2=0.03):

  with ops.name_scope(None, 'SSIM', [img1, img2]):
    # Convert to tensor if needed.
    img1 = ops.convert_to_tensor(img1, name='img1')
    img2 = ops.convert_to_tensor(img2, name='img2')
    # Shape checking.
    _, _, checks = _verify_compatible_image_shapes1(img1, img2)
    with ops.control_dependencies(checks):
      img1 = array_ops.identity(img1)

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = math_ops.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, dtype=tf.float32)
    img1 = tf.image.convert_image_dtype(img1, dtype=tf.float32)
    img2 = tf.image.convert_image_dtype(img2, dtype=tf.float32)
    ssim_per_channel, _ = _ssim_per_channel_2(img1, img2, max_val, filter_size,
                                            filter_sigma, k1, k2)
    # Compute average over color channels.
    return math_ops.reduce_mean(ssim_per_channel, [-1])


max_val=batch_CT.max()-batch_CT.min()
score = ssim_2(batch_CT, batch_CB, max_val)