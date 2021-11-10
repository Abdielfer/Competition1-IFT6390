#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:41:02 2021

@author: stremblay
"""
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

###   Import Corrected  Data ''' To avoid correction avery time we call for debug or train a new model
train_Set = pd.read_csv("train_X_ready.csv", index_col = None)## full data normalized

reduce_Set = pd.read_csv("correctedTrainSet_1.csv", index_col = None)## full data normalized

train_Set.head().to_csv("train_Set_head.csv")

reduce_Set.head().to_csv("train_Set_head.csv")



