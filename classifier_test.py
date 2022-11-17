# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:09:27 2022

@author: vince


"""
import numpy as np
import os
import torch as th
import pandas as pd


dir = "C:/Users/vince/OneDrive/Unibe/Semester_3/diabetes management/project/DMMSR_Dataset/DMMSR_Dataset/adolecents"
data = pd.read_csv(os.path.join(dir, "adolescent#001.csv"))
#data = pd.read.csv(C:/Users/vince/OneDrive/Unibe/Semester_3/diabetes management/project")
insulin = data["sqInsulinNormalBolus"]
insulin = insulin.to_numpy()
time = data["minutesPastSimStart"]
time = time.to_numpy()
cgm = data["cgm"]
cgm = cgm.to_numpy()