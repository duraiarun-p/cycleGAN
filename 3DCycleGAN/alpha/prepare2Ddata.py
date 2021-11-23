#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:54:38 2021

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
import matplotlib.pyplot as plt
import os
import sys, getopt
import scipy.misc
from PIL import Image


# os.environ['CUDA_VISIBLE_DEVICES'] = "" # comment this line when running in eddie
import tensorflow as tf
# import tensorflow_addons as tfa

from tensorflow.keras.utils import Sequence

# cfg = tf.compat.v1.ConfigProto() 
# cfg.gpu_options.allow_growth = True
# sess= tf.compat.v1.Session(config=cfg)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
#%%
datapath='/home/arun/Documents/PyWSPrecision/datasets/printoutslices'

trainCT_path = os.path.join(datapath, 'trainCT')
trainCB_path = os.path.join(datapath, 'trainCB')
testCT_path = os.path.join(datapath, 'validCT')
testCB_path = os.path.join(datapath, 'validCB')

def create_image_array(image_list, image_path):
    image_array = []
    for image_name in image_list:
        image = np.array(Image.open(os.path.join(image_path, image_name)))
        image = image[:, :, np.newaxis]
        image_array.append(image)
    return np.array(image_array)

def loadprintout(trainCT_path,trainCB_path,testCT_path,testCB_path):
    trainCT_image_names = os.listdir(trainCT_path)
    trainCB_image_names = os.listdir(trainCB_path)
    testCT_image_names = os.listdir(testCT_path)
    testCB_image_names = os.listdir(testCB_path)              
    trainCT_images = create_image_array(trainCT_image_names, trainCT_path)
    trainCB_images = create_image_array(trainCB_image_names, trainCB_path)
    testCT_images = create_image_array(testCT_image_names, testCT_path)
    testCB_images = create_image_array(testCB_image_names, testCB_path)
    return {"trainCT_images": trainCT_images, "trainCB_images": trainCB_images,
                "testCT_images": testCT_images, "testCB_images": testCB_images,
                "trainCT_image_names": trainCT_image_names,
                "trainCB_image_names": trainCB_image_names,
                "testCT_image_names": testCT_image_names,
                "testCB_image_names": testCB_image_names}

def create_image_array_gen(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image
                image = np.array(Image.open(os.path.join(image_path, image_name)))
                image = image[:, :, np.newaxis]
            else:                   # RGB image
                image = np.array(Image.open(os.path.join(image_path, image_name)))
            # image = normalize_array_max(image)
            image_array.append(image)

class data_sequence(Sequence):
    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size=1):
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        for image_name in image_list_A:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))
    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))
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
        real_images_A = create_image_array_gen(batch_A, '', 1)
        real_images_B = create_image_array_gen(batch_B, '', 1)
        return real_images_A, real_images_B  # input_data, target_data

def loadprintoutgen(trainCT_path,trainCB_path,batch_size):
    trainCT_image_names = os.listdir(trainCT_path)
    trainCB_image_names = os.listdir(trainCB_path)
    return data_sequence(trainCT_path, trainCB_path, trainCT_image_names, trainCB_image_names, batch_size=batch_size)

data=loadprintout(trainCT_path,trainCB_path,testCT_path,testCB_path)
data_from_gen=loadprintoutgen(trainCT_path, trainCB_path, batch_size=1)


#%%  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)