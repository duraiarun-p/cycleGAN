#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 18:55:12 2021

@author: arun
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


#%%
def tfssim(img1,img2,max_val,filter_size,filter_sigma):
    score = tf.image.ssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return score
def tfmsssim(img1,img2,max_val,filter_size,filter_sigma):
    score = tf.image.ssim_multiscale(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return score
#%%



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

def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):

  c1 = (k1 * max_val)**2
  c2 = (k2 * max_val)**2

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
  num1 = reducer(x * y) * 2.0
  den1 = reducer(math_ops.square(x) + math_ops.square(y))
  c2 *= compensation
  cs = (num1 - num0 + c2) / (den1 - den0 + c2)

  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs

def _ssim_per_channel(img1,
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
    y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    return array_ops.reshape(
        y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0))

  luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1,
                               k2)

  # Average over the second and the third from the last: height, width.
  axes = constant_op.constant([-3, -2], dtype=tf.int32)
  ssim_val = math_ops.reduce_mean(luminance * cs, axes)
  cs = math_ops.reduce_mean(cs, axes)
  return ssim_val, luminance * cs, cs


def tfssim_custom(img1,
         img2,
         max_val,
         filter_size=11,
         filter_sigma=1.5,
         k1=0.01,
         k2=0.03):

  with ops.name_scope(None, 'SSIM2', [img1, img2]):
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
    ssim_per_channel, smap, _ = _ssim_per_channel(img1, img2, max_val, filter_size,
                                            filter_sigma, k1, k2)
    # Compute average over color channels.
    return math_ops.reduce_mean(ssim_per_channel, [-1]), smap

#%%

def _ssim_helper_4_c(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):

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
  
  num1 = reducer(x * y) * 2.0
  den1 = reducer(math_ops.square(x) + math_ops.square(y))
  c2 *= compensation
  cs = (num1 - num0 + c2) / (den1 - den0 + c2)
  
  smap_p=luminance * cs
  
  TH1=0.12*tf.math.reduce_max(d_gm1)
  TH2=0.06*tf.math.reduce_max(d_gm2)
  
  # TH1=0.4*tf.math.reduce_max(d_gm1)
  # TH2=0.2*tf.math.reduce_max(d_gm2)
  
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

def _ssim_per_channel_4_c(img1,
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
  compensation = 1.0 - tf.reduce_sum(tf.square(kernel))

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

  luminance, cs, smap = _ssim_helper_4_c(img1, img2, reducer, max_val, compensation, k1,
                               k2)

  # Average over the second and the third from the last: height, width.
  axes = constant_op.constant([-3, -2], dtype=tf.int32)
  # ssim_map = luminance * cs
  # ssim_val = math_ops.reduce_mean(luminance * cs, axes)
  ssim_val = math_ops.reduce_mean(smap, axes)
  # ssim_val = math_ops.reduce_mean(smap)
  cs = math_ops.reduce_mean(cs, axes)
  return ssim_val, smap, cs


def tfssim4c(img1,
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
    ssim_per_channel, smap, _ = _ssim_per_channel_4_c(img1, img2, max_val, filter_size,
                                            filter_sigma, k1, k2)
    # Compute average over color channels.
    return math_ops.reduce_mean(ssim_per_channel, [-1]),smap

#%%

def _ssim_helper_4_c_gm(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):

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
  # cs = (num1 - num0 + c2) / (den1 - den0 + c2)
  cs = (num1 + c2) / (den1 + c2)
  
  smap_p=luminance * cs
  
  # TH1=0.12*tf.math.reduce_max(d_gm1)
  # TH2=0.06*tf.math.reduce_max(d_gm2)
  
  TH1=0.12*tf.math.reduce_max(d_gm1)
  TH2=0.03*tf.math.reduce_max(d_gm2)
  
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
  # smap=smapR1+smapR2+smapR3+smapR4

  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs, smap

def _ssim_per_channel_4_c_gm(img1,
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
  compensation = 1.0 - tf.reduce_sum(tf.square(kernel))

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

  luminance, cs, smap = _ssim_helper_4_c_gm(img1, img2, reducer, max_val, compensation, k1,
                               k2)

  # Average over the second and the third from the last: height, width.
  axes = constant_op.constant([-3, -2], dtype=tf.int32)
  # ssim_map = luminance * cs
  # ssim_val = math_ops.reduce_mean(luminance * cs, axes)
  ssim_val = math_ops.reduce_mean(smap, axes)
  cs = math_ops.reduce_mean(cs, axes)
  return ssim_val, smap, cs


def tfssim4cg(img1,
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
    ssim_per_channel, smap, _ = _ssim_per_channel_4_c_gm(img1, img2, max_val, filter_size,
                                            filter_sigma, k1, k2)
    # Compute average over color channels.
    return math_ops.reduce_mean(ssim_per_channel, [-1]), smap

_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

def tfmssim_custom(img1,
                    img2,
                    max_val,
                    power_factors=_MSSSIM_WEIGHTS,
                    filter_size=11,
                    filter_sigma=1.5,
                    k1=0.01,
                    k2=0.03):
  with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
    # Convert to tensor if needed.
    img1 = ops.convert_to_tensor(img1, name='img1')
    img2 = ops.convert_to_tensor(img2, name='img2')
    # Shape checking.
    shape1, shape2, checks = _verify_compatible_image_shapes1(img1, img2)
    with ops.control_dependencies(checks):
      img1 = array_ops.identity(img1)

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = math_ops.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, tf.float32)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)

    imgs = [img1, img2]
    shapes = [shape1, shape2]

    # img1 and img2 are assumed to be a (multi-dimensional) batch of
    # 3-dimensional images (height, width, channels). `heads` contain the batch
    # dimensions, and `tails` contain the image dimensions.
    heads = [s[:-3] for s in shapes]
    tails = [s[-3:] for s in shapes]

    divisor = [1, 2, 2, 1]
    divisor_tensor = constant_op.constant(divisor[1:], dtype=tf.int32)

    def do_pad(images, remainder):
      padding = array_ops.expand_dims(remainder, -1)
      padding = array_ops.pad(padding, [[1, 0], [1, 0]])
      return [array_ops.pad(x, padding, mode='SYMMETRIC') for x in images]

    mcs = []
    for k in range(len(power_factors)):
      with ops.name_scope(None, 'Scale%d' % k, imgs):
        if k > 0:
          # Avg pool takes rank 4 tensors. Flatten leading dimensions.
          flat_imgs = [
              array_ops.reshape(x, array_ops.concat([[-1], t], 0))
              for x, t in zip(imgs, tails)
          ]

          remainder = tails[0] % divisor_tensor
          need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0))
          # pylint: disable=cell-var-from-loop
          padded = control_flow_ops.cond(need_padding,
                                         lambda: do_pad(flat_imgs, remainder),
                                         lambda: flat_imgs)
          # pylint: enable=cell-var-from-loop

          downscaled = [
              nn_ops.avg_pool(
                  x, ksize=divisor, strides=divisor, padding='VALID')
              for x in padded
          ]
          tails = [x[1:] for x in array_ops.shape_n(downscaled)]
          imgs = [
              array_ops.reshape(x, array_ops.concat([h, t], 0))
              for x, h, t in zip(downscaled, heads, tails)
          ]

        # Overwrite previous ssim value since we only need the last one.
        ssim_per_channel, _, cs = _ssim_per_channel(
            *imgs,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        # mcs.append(nn_ops.relu(cs))
        mcs.append(ssim_per_channel)

    # Remove the cs score for the last scale. In the MS-SSIM calculation,
    # we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
    mcs.pop()  # Remove the cs score for the last scale.
    # mcs_and_ssim = array_ops.stack(
    #     mcs + [nn_ops.relu(ssim_per_channel)], axis=-1)
    # # Take weighted geometric mean across the scale axis.
    # ms_ssim = math_ops.reduce_prod(
    #     math_ops.pow(mcs_and_ssim, power_factors), [-1])
    ms_ssim=math_ops.reduce_mean(mcs)

    # return math_ops.reduce_mean(ms_ssim, [-1])  # Avg over color channels.
    return ms_ssim

def tfmssim_4c(img1,
                    img2,
                    max_val,
                    power_factors=_MSSSIM_WEIGHTS,
                    filter_size=11,
                    filter_sigma=1.5,
                    k1=0.01,
                    k2=0.03):
  with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
    # Convert to tensor if needed.
    img1 = ops.convert_to_tensor(img1, name='img1')
    img2 = ops.convert_to_tensor(img2, name='img2')
    # Shape checking.
    shape1, shape2, checks = _verify_compatible_image_shapes1(img1, img2)
    with ops.control_dependencies(checks):
      img1 = array_ops.identity(img1)

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = math_ops.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, tf.float32)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)

    imgs = [img1, img2]
    shapes = [shape1, shape2]

    # img1 and img2 are assumed to be a (multi-dimensional) batch of
    # 3-dimensional images (height, width, channels). `heads` contain the batch
    # dimensions, and `tails` contain the image dimensions.
    heads = [s[:-3] for s in shapes]
    tails = [s[-3:] for s in shapes]

    divisor = [1, 2, 2, 1]
    divisor_tensor = constant_op.constant(divisor[1:], dtype=tf.int32)

    def do_pad(images, remainder):
      padding = array_ops.expand_dims(remainder, -1)
      padding = array_ops.pad(padding, [[1, 0], [1, 0]])
      return [array_ops.pad(x, padding, mode='SYMMETRIC') for x in images]

    mcs = []
    for k in range(len(power_factors)):
      with ops.name_scope(None, 'Scale%d' % k, imgs):
        if k > 0:
          # Avg pool takes rank 4 tensors. Flatten leading dimensions.
          flat_imgs = [
              array_ops.reshape(x, array_ops.concat([[-1], t], 0))
              for x, t in zip(imgs, tails)
          ]

          remainder = tails[0] % divisor_tensor
          need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0))
          # pylint: disable=cell-var-from-loop
          padded = control_flow_ops.cond(need_padding,
                                         lambda: do_pad(flat_imgs, remainder),
                                         lambda: flat_imgs)
          # pylint: enable=cell-var-from-loop

          downscaled = [
              nn_ops.avg_pool(
                  x, ksize=divisor, strides=divisor, padding='VALID')
              for x in padded
          ]
          tails = [x[1:] for x in array_ops.shape_n(downscaled)]
          imgs = [
              array_ops.reshape(x, array_ops.concat([h, t], 0))
              for x, h, t in zip(downscaled, heads, tails)
          ]

        # Overwrite previous ssim value since we only need the last one.
        ssim_per_channel, _, cs = _ssim_per_channel_4_c(
            *imgs,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        # mcs.append(nn_ops.relu(cs))
        mcs.append(ssim_per_channel)

    # Remove the cs score for the last scale. In the MS-SSIM calculation,
    # we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
    mcs.pop()  # Remove the cs score for the last scale.
    # mcs_and_ssim = array_ops.stack(
    #     mcs + [nn_ops.relu(ssim_per_channel)], axis=-1)
    # # Take weighted geometric mean across the scale axis.
    # ms_ssim = math_ops.reduce_prod(
    #     math_ops.pow(mcs_and_ssim, power_factors), [-1])
    ms_ssim=math_ops.reduce_mean(mcs)

    # return math_ops.reduce_mean(ms_ssim, [-1])  # Avg over color channels.
    return ms_ssim
#%%
def tfmssim_4cg(img1,
                    img2,
                    max_val,
                    power_factors=_MSSSIM_WEIGHTS,
                    filter_size=11,
                    filter_sigma=1.5,
                    k1=0.01,
                    k2=0.03):
  with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
    # Convert to tensor if needed.
    img1 = ops.convert_to_tensor(img1, name='img1')
    img2 = ops.convert_to_tensor(img2, name='img2')
    # Shape checking.
    shape1, shape2, checks = _verify_compatible_image_shapes1(img1, img2)
    with ops.control_dependencies(checks):
      img1 = array_ops.identity(img1)

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = math_ops.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, tf.float32)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)

    imgs = [img1, img2]
    shapes = [shape1, shape2]

    # img1 and img2 are assumed to be a (multi-dimensional) batch of
    # 3-dimensional images (height, width, channels). `heads` contain the batch
    # dimensions, and `tails` contain the image dimensions.
    heads = [s[:-3] for s in shapes]
    tails = [s[-3:] for s in shapes]

    divisor = [1, 2, 2, 1]
    divisor_tensor = constant_op.constant(divisor[1:], dtype=tf.int32)

    def do_pad(images, remainder):
      padding = array_ops.expand_dims(remainder, -1)
      padding = array_ops.pad(padding, [[1, 0], [1, 0]])
      return [array_ops.pad(x, padding, mode='SYMMETRIC') for x in images]

    mcs = []
    for k in range(len(power_factors)):
      with ops.name_scope(None, 'Scale%d' % k, imgs):
        if k > 0:
          # Avg pool takes rank 4 tensors. Flatten leading dimensions.
          flat_imgs = [
              array_ops.reshape(x, array_ops.concat([[-1], t], 0))
              for x, t in zip(imgs, tails)
          ]

          remainder = tails[0] % divisor_tensor
          need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0))
          # pylint: disable=cell-var-from-loop
          padded = control_flow_ops.cond(need_padding,
                                         lambda: do_pad(flat_imgs, remainder),
                                         lambda: flat_imgs)
          # pylint: enable=cell-var-from-loop

          downscaled = [
              nn_ops.avg_pool(
                  x, ksize=divisor, strides=divisor, padding='VALID')
              for x in padded
          ]
          tails = [x[1:] for x in array_ops.shape_n(downscaled)]
          imgs = [
              array_ops.reshape(x, array_ops.concat([h, t], 0))
              for x, h, t in zip(downscaled, heads, tails)
          ]

        # Overwrite previous ssim value since we only need the last one.
        ssim_per_channel, _, cs = _ssim_per_channel_4_c_gm(
            *imgs,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        # mcs.append(nn_ops.relu(cs))
        mcs.append(ssim_per_channel)

    # Remove the cs score for the last scale. In the MS-SSIM calculation,
    # we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
    mcs.pop()  # Remove the cs score for the last scale.
    # mcs_and_ssim = array_ops.stack(
    #     mcs + [nn_ops.relu(ssim_per_channel)], axis=-1)
    # # Take weighted geometric mean across the scale axis.
    # ms_ssim = math_ops.reduce_prod(
    #     math_ops.pow(mcs_and_ssim, power_factors), [-1])
    ms_ssim=math_ops.reduce_mean(mcs)

    # return math_ops.reduce_mean(ms_ssim, [-1])  # Avg over color channels.
    return ms_ssim