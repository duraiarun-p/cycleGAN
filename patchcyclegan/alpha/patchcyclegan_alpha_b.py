#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:43:40 2021

@author: arun
"""
import tensorflow as tf
from tensorflow import keras
# import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
#%%

CTtrainpathval='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/trainCT/*.mat'
CBtrainpathval='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/trainCB/*.mat'
CTtestpathval='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/testCT/*.mat'
CBtestpathval='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/testCB/*.mat'

# trainCT = tf.data.Dataset.list_files(CTtrainpathval)
# trainCB = tf.data.Dataset.list_files(CBtrainpathval)
# testCT = tf.data.Dataset.list_files(CTtrainpathval)
# testCB = tf.data.Dataset.list_files(CBtrainpathval)

# train_dataset = tf.data.Dataset.from_tensor_slices((trainCT, trainCB))



class CycleGAN(keras.Model):
    
    def loadpatch(self):
        onlyfiles = [f for f in listdir(self.trainCTpath) if isfile(join(self.trainCTpath, f))]
        matfiles=[join(self.trainCTpath,f) for f in onlyfiles]
        mat_fname_ind=np.random.choice(len(matfiles),replace=False)  
        mat_contents=loadmat(matfiles[mat_fname_ind])
        CT_b=mat_contents['CT_b']
        onlyfiles = [f for f in listdir(self.trainCBpath) if isfile(join(self.trainCBpath, f))]
        matfiles=[join(self.trainCBpath,f) for f in onlyfiles]
        mat_fname_ind=np.random.choice(len(matfiles),replace=False)  
        mat_contents=loadmat(matfiles[mat_fname_ind])
        CB_b=mat_contents['CB_b']
        yield tf.convert_to_tensor(CT_b, dtype=tf.float32), tf.convert_to_tensor(CB_b, dtype=tf.float32)
    
    
    def get_ds(self):
        return tf.data.Dataset.from_generator(
            self.loadpatch,
            output_signature=(tf.TensorSpec(shape=(32, 32, 32), dtype=tf.float32), tf.TensorSpec(shape=(32, 32, 32), dtype=tf.float32))
            )
    
    def __init__(self,CTpath,CBpath):
        self.trainCTpath=CTpath
        self.trainCBpath=CBpath
        

CTpath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/trainCT/'
CBpath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/trainCB/'

batch_size = 10
buffer_size = 10*batch_size
CG=CycleGAN(CTpath, CBpath)
train_ds = CG.get_ds()
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.shuffle(buffer_size)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)