#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:52:32 2021

@author: arun
"""

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
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
import os
import sys, getopt
import scipy.io


# os.environ['CUDA_VISIBLE_DEVICES'] = "" # comment this line when running in eddie
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

# import scipy
from tensorflow.keras.utils import Sequence

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
#%% Data loader using data generator

    
DataPath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db1/'
trainCT_path = os.path.join(DataPath, 'trainCT')
trainCB_path = os.path.join(DataPath, 'trainCB')
trainCT_image_names = os.listdir(trainCT_path)
trainCB_image_names = os.listdir(trainCB_path)

# mat_contents=h5py.File(os.path.join(trainCT_path,trainCT_image_names[0]),'r')
mat_contents=scipy.io.loadmat(os.path.join(trainCT_path,trainCT_image_names[0]))
CT_b=mat_contents['CT_b']
CT_tf=tf.convert_to_tensor(CT_b,dtype=tf.float64)
CT_tf=tf.expand_dims(CT_tf, axis=-1)

def create_image_array_gen_CT(trainCT_image_names, trainCT_path):
    image_array = []
    for image_name in trainCT_image_names:
        mat_contents=scipy.io.loadmat(os.path.join(trainCT_path,image_name))
        CT_tf=mat_contents['CT_b']
        # CT_tf=tf.convert_to_tensor(CT_b,dtype=tf.float64)
        CT_tf=tf.expand_dims(CT_tf, axis=-1)
        image_array.append(CT_tf)
    return np.array(image_array)

def create_image_array_gen_CB(trainCT_image_names, trainCT_path):
    image_array = []
    for image_name in trainCT_image_names:
        mat_contents=scipy.io.loadmat(os.path.join(trainCT_path,image_name))
        CT_tf=mat_contents['CB_b']
        # CT_tf=tf.convert_to_tensor(CT_b,dtype=tf.float64)
        CT_tf=np.expand_dims(CT_tf, axis=-1)
        image_array.append(CT_tf)
    return np.array(image_array)

class data_sequence(Sequence):
    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size):
        # self.newshape=newshape
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        for image_name in image_list_A:
            # if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            # if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))
    
    def __len__(self):
        # no=1
        return int(min(len(self.train_A), len(self.train_B)) / float(self.batch_size))
        # return int(no)
    
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
                
#%%
def loadprintoutgen(trainCT_path,trainCB_path,batch_size):
    trainCT_image_names = os.listdir(trainCT_path)
    trainCB_image_names = os.listdir(trainCB_path)
    # return trainCT_image_names,trainCB_image_names
    return data_sequence(trainCT_path, trainCB_path, trainCT_image_names, trainCB_image_names,batch_size=batch_size)

data=loadprintoutgen(trainCT_path,trainCB_path,batch_size=4)
# trainCT_image_names,trainCB_image_names=loadprintoutgen(trainCT_path,trainCB_path,batch_size=1)                

# trainCT_image_names = os.listdir(trainCT_path)
# trainCB_image_names = os.listdir(trainCB_path)


# data=data_sequence(trainCT_path, trainCB_path, trainCT_image_names, trainCB_image_names,batch_size=1)

# CT_image_array=create_image_array_gen_CT(trainCT_image_names, trainCT_path)
# CB_image_array=create_image_array_gen_CB(trainCB_image_names, trainCB_path)
for image in data:
    batch_CT=image[0]
    batch_CB=image[1]