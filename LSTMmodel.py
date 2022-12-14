# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 10:12:37 2022

@author: vince
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from numpy import array
import os
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
 
 series_length = 15
 
 data, labels = split_data(cgm,insulin,series_length)
 labels = labels.astype(int)
 
 for row in range(0,len(data)):
     mue = np.mean(data[row])
     sigma = np.std(data[row])
     data[row] = (data[row]-mue)/sigma
     

 
 data_train = data[0:int(0.8*len(data))]
 labels_train = labels[0:int(0.8*len(data))]
 
 data_val = data[int(0.8*len(data)):int(0.9*len(data))]
 labels_val = labels[int(0.8*len(data)):int(0.9*len(data))]
 
 data_test = data[int(0.9*len(data)):-1]
 labels_test = labels[int(0.9*len(data)):-1]
 
 #normalization

 

#model
n_steps = series_length
n_features = 1
model = Sequential()
model.add(LSTM(25, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
#print("test")

from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss
training_generator = BalancedBatchGenerator(
    data_train, labels_train, sampler=NearMiss(), batch_size=10)



model.fit(data_train,labels_train,epochs=2000)
predict = model.predict(data_val)

# from sklearn.metrics import accuracy_score,precision_score,recall_score
# from imblearn.metrics import specificity_score


# acc = accuracy_score(labels_val,predict)
# precision = precision_score(labels_val,predict_labels)
# recall = recall_score(labels_val,predict)
# spec = specificity_score(labels_val, predict)


from sklearn.metrics import roc_curve

fpr, tpr,tresholds = roc_curve(labels_val,predict) #fpr = false positive, trp = true positive
#select threshold from validation set and apply to test set (10% val set, train 70%, rest test)

plt.subplot(1,2,1)
plt.plot(fpr,tpr)
plt.title("ROC_curve")
plt.xlabel("specifity")
plt.ylabel("sensitivity")
plt.grid()

from sklearn.metrics import precision_recall_curve

prec,rec,tresholds2 = precision_recall_curve(labels_val,predict)
plt.subplot(1,2,2)
plt.plot(rec,prec)
plt.title("precision-recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()

from sklearn.metrics import accuracy_score,precision_score,recall_score
#DICE and housedorf etc...

