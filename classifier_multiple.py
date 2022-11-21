# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:09:27 2022

@author: vince


"""
import glob

import numpy as np
import os
#import torch as th
import pandas as pd
import joblib


PROJECT_HOMEFOLDER=os.getcwd()
PATH_TO_DMMSR_DATASET=PROJECT_HOMEFOLDER+"\DMMSR_Dataset\DMMSR_Dataset"

for filepath in glob.glob(PATH_TO_DMMSR_DATASET + "/**/*.csv"):
    filename=os.path.basename(filepath)
    file = pd.read_csv(filepath)
    insulin = file["sqInsulinNormalBolus"].to_numpy()
    time = file["minutesPastSimStart"].to_numpy()
    cgm = file["cgm"].to_numpy()

    interval = 15  # choose an intervaöö of 15 minutes
    length = int(np.floor(np.size(time) / interval))

    labels = []

    for i in range(0, length):
        labels.append(any(insulin[i * interval:i * interval + interval] != 0))

    labels = (np.asarray(labels)).astype(float)  # 2976x1 float array

    data = []

    for i in range(0, length):
        data.append(cgm[i * interval:i * interval + interval])

    data = np.asarray(data)  # 2976x15 float array

    joblib.dump(labels, 'labels_' + filename)
    joblib.dump(data, 'data_' + filename)

pass
#classifier

