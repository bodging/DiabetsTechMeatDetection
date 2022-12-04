# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 10:12:37 2022

@author: vince
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from numpy import array
import os
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
 X = list()
 y = []
 for i in range(0,len(sequence)):
 # find the end of this pattern
     end_ix = i + n_steps
 # check if we are beyond the sequence
     if end_ix > len(sequence)-1:
        break
 # gather input and output parts of the pattern
     seq_x = sequence[i:end_ix]
     X.append(seq_x)
     y.append(any(seq_x!=0))
 return array(X), array(y)

def split_data(cgm,insulin, n_steps):
 data  = []
 labels = []
 for i in range(0,len(cgm)):
 # find the end of this pattern
     end_ix = i + n_steps
 # check if we are beyond the sequence
     if end_ix > len(cgm)-1:
        break
 # gather input and output parts of the pattern
     seq_x = cgm[i:end_ix]
     data.append(seq_x)
     seq_y = any(insulin[i:end_ix]!=0)
     labels.append(seq_y)
 return array(data), array(labels)

raw_seq = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
# choose a number of time steps
n_steps = 3
# split into samples
lenght = 3


X, y = split_sequence(raw_seq, n_steps)
y = y.astype(int)
# summarize the data
for i in range(len(X)):
 print(X[i], y[i])
 
 # read data
 dir = "C:/Users/vince/OneDrive/Unibe/Semester_3/diabetes management/project/DMMSR_Dataset/DMMSR_Dataset/adolecents"
 file = pd.read_csv(os.path.join(dir, "adolescent#001.csv"))
 insulin = file["sqInsulinNormalBolus"].to_numpy()
 time = file["minutesPastSimStart"].to_numpy()
 cgm = file["cgm"].to_numpy()
 
 data, labels = split_data(cgm,insulin,15)
 labels = labels.astype(int)



#model
n_steps = 15
n_features = 1
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
#print("test")
