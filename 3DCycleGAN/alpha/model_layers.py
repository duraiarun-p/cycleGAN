#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:16:40 2021

@author: arun
"""

from tensorflow import keras
from tensorflow.keras import layers

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

def build_generator3D(input_layer_shape):
    ipL=keras.Input(shape=input_layer_shape,name='Input')
    opL1=convblk3d(ipL,genafilter,kernel_size,stride2)
    opL2=convblk3d(opL1,genafilter*2,kernel_size,stride2)
    opL3=convblk3d(opL2,genafilter*4,kernel_size,stride2)
    opL4=layers.Conv3D(filters=genafilter*8, kernel_size=kernel_size, strides=stride2,padding='SAME')(opL3)
    # opL4=self.convblk3d(opL3,self.genafilter*8,self.kernel_size,self.stride2)
    
    opL5=attentionblk3D(opL3,opL4,filters=self.genafilter*8, kernel_size=self.kernel_size, strides=self.stride2)
    
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

def build_discriminator3D(input_layer_shape):
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