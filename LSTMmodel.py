# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 10:12:37 2022

@author: vince
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from numpy import array
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from keras import backend as K

#functions
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def split_data(cgm,insulin, n_steps):
 data  = []
 labels = []
 for i in range(0,len(cgm)-100):
 # find the end of this pattern
     end_ix = i + n_steps
 # check if we are beyond the sequence
     if end_ix > len(cgm)-1:
        break
     offset = 15
     i_off = i+offset
     end_ix_off = end_ix+offset
 
     seq_x = cgm[i_off:end_ix_off]
     data.append(seq_x)
     seq_y = any(insulin[i:end_ix]!=0) #to detect a jump in insulin
     labels.append(seq_y)
 return array(data), array(labels)

 
 # read data
dir = "C:/Users/vince/OneDrive/Unibe/Semester_3/diabetes management/project/DMMSR_Dataset/DMMSR_Dataset/adolecents"
file = pd.read_csv(os.path.join(dir, "adolescent#005.csv"))
insulin = file["sqInsulinNormalBolus"].to_numpy()
time = file["minutesPastSimStart"].to_numpy()
cgm = file["cgm"].to_numpy()
cho = file["CR"].to_numpy()


#create samples 
series_length = 30
data, labels = split_data(cgm,insulin,series_length)
labels = labels.astype(int)

 #normalization
 
for row in range(0,len(data)):
    mue = np.mean(data[row])
    sigma = np.std(data[row])
    data[row] = (data[row]-mue)/sigma
     

#create sets
 
data_train = data[0:int(0.8*len(data))]
labels_train = labels[0:int(0.8*len(data))]

 
data_val = data[int(0.8*len(data)):int(0.9*len(data))]
labels_val = labels[int(0.8*len(data)):int(0.9*len(data))]
 
data_test = data[int(0.9*len(data)):-1]
labels_test = labels[int(0.9*len(data)):-1]
 

 

#model
n_steps = series_length
n_features = 1
model = Sequential()
model.add(LSTM(25, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])


#minoritiy oversampling
from imblearn.over_sampling import SMOTE
sm = SMOTE( sampling_strategy=1, random_state=None, k_neighbors=5, n_jobs=None)
x_resampled, y_resampled = sm.fit_resample(data_train, labels_train)

#training
model.fit(x_resampled,y_resampled,batch_size=16, epochs=10)
#model.fit(data_train,labels_train,batch_size=16, epochs=1)

#prediction
predict = model.predict(data_val)
predict_test = model.predict(data_test)




#evaluation
from sklearn.metrics import roc_curve

fpr, tpr,tresholds = roc_curve(labels_val,predict) #fpr = false positive, trp = true positive
#select threshold from validation set and apply to test set (10% val set, train 70%, rest test)
fpr_t, tpr_t,tresholds_t = roc_curve(labels_test,predict_test)

plt.subplot(1,2,1)
plt.plot(fpr,tpr,label="validation")
plt.plot(fpr_t,tpr_t,'r',label="testing")
plt.title("ROC_curve")
plt.xlabel("specifity")
plt.ylabel("sensitivity")
plt.grid()

from sklearn.metrics import precision_recall_curve

prec,rec,tresholds2 = precision_recall_curve(labels_val,predict)
prec_t,rec_t,tresholds2_t = precision_recall_curve(labels_test,predict_test)

plt.subplot(1,2,2)
plt.plot(rec,prec,label="validation")
plt.plot(rec_t,prec_t,'r',label="testing")
plt.title("precision-recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


ROC_area = roc_auc_score(labels_val,predict)

avg_prec = average_precision_score(labels_val,predict)
print('area under ROC:', ROC_area, 'average precision:', avg_prec)

#chosing good tresholds:
#tresholds were true positive - false positive is maximal
opt_index = np.argmax(tpr-fpr)
opt_treshold = tresholds[opt_index]

opt_index2 = np.argmin(abs(prec-rec))

opt_treshold2 = tresholds2[opt_index2]
print('optimal teshold is:', opt_treshold)

from sklearn.metrics import accuracy_score

predict_tresh = (predict>=opt_treshold)
predict_tresh2 = (predict>=opt_treshold2)

acc = accuracy_score(labels_val,predict_tresh2)

spec_tresh = 1-fpr[opt_index]
sens_tresh = tpr[opt_index]
prec_tresh = prec[opt_index2]
rec_tresh = rec[opt_index2]

print('accuracy=', acc, 'specifity=',spec_tresh,' sensitivty=',sens_tresh)
print('precision =', prec_tresh, ' recall =', rec_tresh)

predict_bin = predict_tresh2
labels_val = labels_val.reshape(predict_bin.shape)
diff=(abs(predict_bin-labels_val))
1-np.sum(diff)/len(diff)




        



