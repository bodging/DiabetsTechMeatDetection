# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:09:27 2022

@author: vince


"""
import numpy as np
import os
import torch as th
import pandas as pd
import matplotlib.pyplot as plt

#functions

def logistic_regression (x: th.tensor, weights: th.tensor, bias) -> th.tensor:
    bias = bias[0:x.shape[0]]
    return th.sigmoid(x@weights+bias)

def binary_cross_entropy(p_hat: th.tensor, y : th.tensor) ->th.tensor:
    case1 = -th.log(1-p_hat)
    case2 = -th.log(p_hat)
    return th.where(y==0,case1,case2)

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
labels = th.from_numpy(labels)

data = []

for i in range(0,length):
    data.append(cgm[i*15:i*15+15])
    
data = np.asarray(data) #2976x15 float array
data = th.from_numpy(data).type(th.FloatTensor)


#classifier
data_train = data[0:int((2/3*length))]
labels_train = labels[0:int((2/3*length))]

data_test = data[int((2/3*length)):-1]
labels_test = labels[int((2/3*length)):-1]


num_samples = data_train.shape[0]
num_dimensions = data_train.shape[1]
weights = th.randn(num_dimensions,1)/num_samples
bias = th.zeros(num_samples,1)
weights.requires_grad_()
bias.requires_grad_()
learning_rate = 10**-10
losses = []
meanLoss = []

num_iterations = 100

print(weights.max())


for i in range(num_iterations):
    weights.requires_grad_()
    bias.requires_grad_()
    loss = 0

    p_hat = logistic_regression(data_train,weights,bias)
    p_hat = p_hat.reshape(labels_train.shape)


    loss = binary_cross_entropy(p_hat, labels_train)
    if(loss.max() == th.inf):
        print('learning rate too high')
        break

    tens = th.ones(loss.shape)
    loss.backward(tens)
    with th.no_grad():
        weights = weights-learning_rate*weights.grad
        bias = bias-learning_rate*bias.grad
    weights.grad = None
    bias.grad = None
    lossi = th.detach(loss)
    losses.append(lossi)
    meanLoss.append(abs(lossi).mean())
    print(weights.max())
    
x_axis = np.linspace(1,num_iterations,num_iterations)
fig, axs = plt.subplots(2)

axs[0].plot(losses[-1])
axs[1].plot(x_axis,meanLoss)

print(losses[-1].mean())
    
# # weights.shape
# data_test.shape
# bias.shape
predict = logistic_regression(data_test,weights,bias)

from sklearn.metrics import roc_curve

fpr, tpr,tresholds = roc_curve(labels_test,predict) #fpr = false positive, trp = true positive

plt.subplot(1,2,1)
plt.plot(fpr,tpr)
plt.title("ROC_curve")
plt.xlabel("specifity")
plt.ylabel("sensitivity")
plt.grid()

from sklearn.metrics import precision_recall_curve

prec,rec,tresholds2 = precision_recall_curve(labels_test,predict)
plt.subplot(1,2,2)
plt.plot(rec,prec)
plt.title("precision-recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


ROC_area = roc_auc_score(labels_test,predict)

avg_prec = average_precision_score(labels_test,predict)
print('area under ROC:', ROC_area, 'average precision:', avg_prec)

#chosing good tresholds:
#tresholds were true positive - false positive is maximal
opt_index = np.argmax(tpr-fpr)
opt_treshold = tresholds[opt_index]

opt_index2 = np.argmin(abs(prec-rec))

opt_treshold2 = tresholds2[opt_index2]
print('optimal teshold is:', opt_treshold)

from sklearn.metrics import accuracy_score

predict_tresh = (predict>=opt_treshold).type(th.uint8)
predict_tresh2 = (predict>=opt_treshold).type(th.uint8)

acc = accuracy_score(labels_test,predict_tresh)
acc2 = accuracy_score(labels_test,predict_tresh2)

spec_tresh = fpr[opt_index]
sens_tresh = tpr[opt_index]
prec_tresh = prec[opt_index2]
rec_tresh = rec[opt_index2]

print('accuracy=', acc, 'specifity=',spec_tresh,' sensitivty=',sens_tresh)
print('precision =', prec_tresh, ' recall =', rec_tresh)
