#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:11:41 2021

@author: arun
"""

# import os
# import zipfile
# import numpy as np
# import tensorflow as tf


# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

from tensorflow import keras
from tensorflow.keras import layers


def get_model(img_shape):
    """Build a 3D convolutional neural network model."""

#If your image batch is of N images of HxW size with C channels, tensorflow uses the NHWC ordering
    inputs = keras.Input(shape=img_shape)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    # x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.GlobalAveragePooling3D()(x)
    # x = layers.Dense(units=512, activation="relu")(x)
    # x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
img_shape=(3,512,512,1)
model = get_model(img_shape)
model.summary()

width=128 
height=128 
depth=64
img_shape1=(width, height, depth, 1)