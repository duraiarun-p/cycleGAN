#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 13:22:31 2021

@author: arun
"""
import numpy as np
class student:
    # Class attribute
    species = "Canis familiaris"
    

    def __init__(self, name, age, marks):
        # instance attributes
        self.name = name
        self.age = age
        # self.marks = marks
        self.marks=np.array(marks)
        Len=np.shape(self.marks)
        self.num_paper=Len
        # self.marksum=self.totalmark(marks)
        
        
    def totalmark(self):
        # call those attributes inside herr to execute the method
        TotMark=np.sum(self.marks)
        return f"{TotMark}"


