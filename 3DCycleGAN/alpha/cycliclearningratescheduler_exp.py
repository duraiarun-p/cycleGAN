#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:28:16 2021

@author: arun
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg

lr_max=0.003
lr=0.0002
fullcyc=50
epochs=200
cycles_N=int(epochs/fullcyc)
epochtime=np.linspace(0, fullcyc,fullcyc)
epochfulltime=np.linspace(0, epochs,epochs)

# sig=[]
sig=np.zeros((epochs,))

for cycle in range(cycles_N):
    lr_pp=lr_max-lr
    sig3=(sg.sawtooth(np.pi*2*epochtime,width=0.5)+1)*lr_pp
    # sig3a=sig3[0:fullcyc//2]
    # sig3b=sig3[fullcyc//2:fullcyc]
    # sig3=np.concatenate((sig3b,sig3a))
    lr_max=lr_max/2
    plt.figure(2),plt.subplot(2,2,cycle+1),plt.plot(epochtime,sig3),plt.show()
    sig[cycle*(fullcyc):cycle*(fullcyc)+fullcyc]=sig3

bs=10/2
TB=np.zeros((epochs,))
for epoch in range(epochs):
    if epoch % 50 ==0:
        bs=round(bs*2)
        
    TB[epoch]=bs
    # print(cycle)
    # print(cycle*(fullcyc))
    # print(cycle*(fullcyc)+fullcyc)
    # sig.append(sig3)

# sig=np.asarray(sig)
#%%
plt.figure(1),plt.plot(epochtime,sig3),plt.show()
plt.figure(3),plt.plot(epochfulltime,sig),plt.show()