#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[2]:

import fase
fase.USE_FPGA = False
from fase.core.heaan import he

cam='a'
action = 6

import numpy as np
import pickle
import torch
#import matplotlib.pyplot as plt 
#from fase.core import heaan
#from typing import List, Callable

from fase import hnrf as hnrf

from fase.hnrf.tree import NeuralTreeMaker
from fase.hnrf import heaan_nrf 
#from fase.hnrf.hetree_nrf import HomomorphicModel 
#import importlib

from time import time


# In[3]:


def decrypt_print(ctx, n=20):
    res1 = decrypt(secretKey, ctx)
    print(res1[:n])
    
def decrypt(secretKey, enc):
    featurized = scheme.decrypt(secretKey, enc)
    arr = np.zeros(n, dtype=np.complex128)
    featurized.__getarr__(arr)
    return arr.real

def encrypt(val):
    ctxt = he.Ciphertext()#logp, logq, n)
    vv = np.zeros(n) # Need to initialize to zero or will cause "unbound"
    vv[:len(val)] = val
    scheme.encrypt(ctxt, he.Double(vv), n, logp, logq)
    del vv
    return ctxt


from torch.utils import data

class TabularDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X: np.ndarray, y: np.ndarray):
        'Initialization'
        self.X, self.y = X,y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = torch.tensor(self.X[index]).float()
        y = torch.tensor(self.y[index])

        return X, y


model_dir = "/home/etri_ai2/work/data/BBS/Trained_model/"
save_dir = "/home/etri_ai2/work/Kinect_BBS_demo/server/models/"

small = True

print("ACTION", action)
print("cam", cam)
if small:
    fn_model = f"trained_model_{action}_{cam}.pickle"
    fn_data = f"BBS_dataset_{action}_{cam}.pickle"

fn_model = model_dir + fn_model
fn_dat = model_dir + fn_data

rf_model = pickle.load(open(fn_model, "rb"))

print("model's depth:", rf_model.max_depth)
print("model's tree count:", rf_model.n_estimators)

dataset = pickle.load(open(fn_dat, "rb"))

X_train = dataset["train_x"]
y_train = dataset["train_y"]
X_valid = dataset["valid_x"]
y_valid = dataset["valid_y"]

print("min max of input dataset")
print(X_train.min(), X_train.max())
print(X_valid.min(), X_valid.max())


# # Convert RF to NRF
# 
# fastai를 안 쓰고싶지만, fastai가 훨씬 빠름.. 일단 사용하기로

# In[86]:


from sklearn.tree import BaseDecisionTree
from fase.hnrf.tree import NeuralRF

dilatation_factor = 10
polynomial_degree = 10

estimators = rf_model.estimators_

my_tm_tanh = NeuralTreeMaker(torch.tanh, 
                            use_polynomial=True,
                            dilatation_factor=dilatation_factor, 
                            polynomial_degree=polynomial_degree)

Nmodel = NeuralRF(estimators, my_tm_tanh)


# In[87]:


# NRF 성능 테스트
with torch.no_grad():
    neural_pred = Nmodel(torch.tensor(X_train).float()).argmax(dim=1).numpy()
    
pred = rf_model.predict(X_train)
print(f"Original accuracy : {(pred == y_train).mean()}")
print(f"Accuracy of tanh : {(neural_pred == y_train).mean()}")
print(f"Match between tanh and original : {(neural_pred == pred).mean()}")


# ## Fine tuning

bs = 128

train_ds = TabularDataset(X_train, y_train)
valid_ds = TabularDataset(X_valid, y_valid)

train_dl = data.DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_dl = data.DataLoader(valid_ds, batch_size=bs)
fix_dl = data.DataLoader(train_ds, batch_size=bs, shuffle=False)


# In[89]:


Nmodel.freeze_layer("comparator")
#Nmodel.freeze_layer("matcher")

for p in Nmodel.parameters():
    print(p.shape, p.requires_grad)


# In[90]:


import fastai
print(fastai.__version__)

from fastai.basic_data import DataBunch
from fastai.tabular.learner import Learner
from fastai.metrics import accuracy

from fase.hnrf.tree import CrossEntropyLabelSmoothing
import torch.nn as nn

data = DataBunch(train_dl, valid_dl, fix_dl=fix_dl)

criterion = CrossEntropyLabelSmoothing()

learn = Learner(data, Nmodel, loss_func=criterion, metrics=accuracy)

learn.fit_one_cycle(100, 0.1)


# In[92]:
learn.fit_one_cycle(100, 0.01)
learn.fit_one_cycle(100, 0.05)
learn.fit_one_cycle(100, 0.005)
learn.fit_one_cycle(100, 0.001)


# ## Fine-tuned NRF model

# In[94]:


pred = rf_model.predict(X_valid)

with torch.no_grad():
    neural_pred = Nmodel(torch.tensor(X_valid).float()).argmax(dim=1).numpy()

print(f"Original accuracy : {(pred == y_valid).mean()}")
print(f"Accuracy : {(neural_pred == y_valid).mean()}")
print(f"Same output : {(neural_pred == pred).mean()}")

# # 0. HEAAN context

# In[95]:


class Param():
    def __init__(self, n=None, logn=None, logp=None, logq=None, logQboot=None):
        self.n = n
        self.logn = logn
        self.logp = logp
        self.logq = logq 
        self.logQboot = logQboot
        if self.logn == None:
            self.logn = int(np.log2(n))


# In[96]:


logq = 540
logp = 30
logn = 14
n = 1*2**logn
slots = n

parms = Param(n=n, logp=logp, logq=logq)

do_reduction=False


ring = he.Ring()
secretKey = he.SecretKey(ring)
scheme = he.Scheme(secretKey, ring, False)

algo = he.SchemeAlgo(scheme)


# reduction때는 right rotation N_class개 필요. 
if do_reduction:
    Nclass = Nmodel.head.shape[0]
    scheme.addLeftRotKeys(secretKey)
    for i in range(Nclass):
        scheme.addRightRotKey(secretKey, i+1) # 
else:
    # reduction 안 하면 하나짜리 rotation만 여러번 반복.
    scheme.addLeftRotKey(secretKey, 1)


from fase.hnrf.hetree import HNRF
h_rf = HNRF(Nmodel)
pickle.dump(Nmodel, open(save_dir+f"Nmodel_{action}_{cam}.pickle", "wb"))
pickle.dump(h_rf, open(save_dir+f"h_rf_{action}_{cam}.pickle", "wb"))
nrf_evaluator = heaan_nrf.HomomorphicTreeEvaluator.from_model(h_rf,
                                                    scheme,
                                                    parms,
                                                    my_tm_tanh.coeffs,
                                                    do_reduction = do_reduction
                                                    )

featurizer = heaan_nrf.HomomorphicTreeFeaturizer(h_rf.return_comparator(), scheme, parms)

for xx, yy in zip(X_valid[:4], y_valid[:4]):
    ctx = featurizer.encrypt(xx)
    t0 = time()
    result = nrf_evaluator(ctx)
    print(f"Took {time() - t0:.2f} seconds")

    pred = []
    pred2 = []
    for res in result:
        dec = decrypt(secretKey, res)
        #print(dec[:120])
        pred.append(np.sum(dec))
        pred2.append(np.sum(dec[:56]))
      
    print(f"Prediction: {np.argmax(pred)} == {yy}?")
    print(f"Prediction2: {np.argmax(pred2)} == {yy}?")
