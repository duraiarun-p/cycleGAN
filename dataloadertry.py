#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:37:17 2021

@author: arun
"""

import numpy as np
import h5py

mat_fname='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB3/ZC001-DB3.mat'
mat_contents=h5py.File(mat_fname)
mat_contents_list=list(mat_contents.keys())

PlanCTDataRef=mat_contents['CTInfoCell']
PlanCTData=PlanCTDataRef.value
pCT=PlanCTData[2]
# PlanCTData=np.transpose(PlanCTData,(2,1,0))