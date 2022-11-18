# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:09:27 2022

@author: vince


"""
import numpy as np
import os
import torch as th
import pandas as pd

#data handling

dir = "C:/Users/vince/OneDrive/Unibe/Semester_3/diabetes management/project/DMMSR_Dataset/DMMSR_Dataset/adolecents"
file = pd.read_csv(os.path.join(dir, "adolescent#001.csv"))
insulin = file["sqInsulinNormalBolus"].to_numpy()
time = file["minutesPastSimStart"].to_numpy()
cgm = file["cgm"].to_numpy()

interval = 15 #choose an intervaöö of 15 minutes
length = int(np.floor(np.size(time)/interval)) 

labels = []

for i in range(0,length):
     labels.append(any(insulin[i*15:i*15+15]!=0))

labels = (np.asarray(labels)).astype(float) #2976x1 float array

data = []

for i in range(0,length):
    data.append(cgm[i*15:i*15+15])
    
data = np.asarray(data) #2976x15 float array

#classifier

