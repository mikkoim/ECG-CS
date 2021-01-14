# -*- coding: utf-8 -*-

import numpy as np
import pickle
import copy
import scipy.io as sio
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from net1d import Net1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary


is_debug = False

batch_size = 32

ds_name = 'mitdb_inverse_sample.mat'
mat = sio.loadmat(ds_name)
print(mat.keys())

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
def read_mat(location, X_key, y_key, standardize=True):
    mat = sio.loadmat(location)

    X0 = np.asarray(mat[X_key])
    y0 = np.asarray(mat[y_key])

    X = np.delete(X0, np.where(y0==5), axis=0)
    y = np.delete(y0, np.where(y0==5), axis=0)

    if standardize:
      scaler = StandardScaler()
      X = scaler.fit_transform(X.T).T

    enc = LabelEncoder()
    y = enc.fit_transform(y.ravel()).ravel()


    print("X shape: ", X.shape)
    print("y shape: ",y.shape)

    return X, y

def split_dataset(X, y):

    # split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # shuffle train
    shuffle_pid = np.random.permutation(y_train.shape[0])
    X_train = X_train[shuffle_pid]
    y_train = y_train[shuffle_pid]

    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    print("Train shape: ", X_train.shape, y_train.shape)
    print("Test shape: ", X_test.shape, y_test.shape)
    print("N classes: ", len(np.unique(y_train)))

    return X_train, X_test, y_train, y_test

## Load model

from resnet1d import ResNet1D

device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
print(device)

X, y = read_mat('mitdb_inverse_sample', 'X', 'y', standardize=True)
X_train,       _,  y_train,      _  = split_dataset(X,y)

n_classes = len(np.unique(y)) # 5

def get_model():
  model = ResNet1D(
      in_channels=1, 
      base_filters=64, # 64 for ResNet1D, 352 for ResNeXt1D
      kernel_size=16, 
      stride=2, 
      groups=1, 
      n_block=48, 
      n_classes=n_classes, 
      downsample_gap=6, 
      increasefilter_gap=12, 
      use_do=True)

  model.to(device)
  return model
base_model = get_model()
summary(base_model, (X_train.shape[1], X_train.shape[2]), device=device_str)

# Training a single model

## TRAIN DATASET
print(ds_name)
dataset = MyDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size)

def my_train(model, dl, n_epoch, lr):

  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
  loss_func = torch.nn.CrossEntropyLoss()

  for _ in tqdm(range(n_epoch), desc="epoch", leave=False):
      running_loss = []
      # train
      model.train()
      for batch_idx, batch in enumerate(dl):

          input_x, input_y = tuple(t.to(device) for t in batch)
          pred = model(input_x)
          loss = loss_func(pred, input_y)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          running_loss.append(loss.item())

      print("\t Loss: {:.4f}".format(np.mean(running_loss)))
      scheduler.step(_)

  return model

base_model = my_train(base_model, dataloader, n_epoch = 10, lr=1e-3)

ds_name = 'frequency_stat_cr065.mat'
print(ds_name)
X, y = read_mat(ds_name, 'X', 'y', standardize=True)
X_train,       X_test,  y_train,      y_test  = split_dataset(X,y)

dataset = MyDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size)

mod = copy.deepcopy(base_model)
mod = my_train(mod, dataloader, n_epoch = 100, lr=1e-3)

# Test
print(ds_name)
X, y = read_mat(ds_name, 'X', 'y', standardize=True)
_,       X_test,  _,      y_test  = split_dataset(X,y)

def evaluate(X,y, model):
  ds = MyDataset(X, y)
  dl = DataLoader(ds, batch_size=batch_size, drop_last=False)

  model.eval()
  all_pred_prob = []
  with torch.no_grad():
      for batch in dl:
          input_x, input_y = tuple(t.to(device) for t in batch)
          pred = model(input_x)
          all_pred_prob.append(pred.cpu().data.numpy())

  all_pred_prob = np.concatenate(all_pred_prob)

  return np.argmax(all_pred_prob, axis=1)

y_pred = evaluate(X_test, y_test, mod)
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
C = confusion_matrix(y_test, y_pred)
print(C)
plt.matshow(C)

# Illustrate successful masking

S, y = read_mat(ds_name, 'S', 'y', standardize=True)
_,       S_test,  _,      y_test  = split_dataset(S,y)

SA, y = read_mat(ds_name, 'SA', 'y', standardize=True)
_,       SA_test,  _,      _  = split_dataset(SA,y)

SB, y = read_mat(ds_name, 'SB', 'y', standardize=True)
_,       SB_test,  _,      _  = split_dataset(SB,y)

S0, _ = read_mat(ds_name, 'S', 'y', standardize=False)
_,       S0_test,  _,      _  = split_dataset(S0,y)

S_nn, y_nn =preprocess_data(S, y)
SA_nn, _ =preprocess_data(SA, y)
SB_nn, _ =preprocess_data(SB, y)

print(np.where(y!=0))

i = 45
print(y[i])

s = S[i,:].ravel()
sA = SA[i,:].ravel()
sB = SB[i,:].ravel()


y_predA = evaluate(SA_nn, y_nn)
y_predB = evaluate(SB_nn, y_nn)

print("True label: ", y[i])
print("sA prediction: ", y_predA[i])
print("sB prediction: ", y_predB[i])

plt.plot(s)
plt.figure()
plt.plot(sA)

"""# Model k-fold evaluation"""

def preprocess_data(X,y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X.T).T

    X = np.expand_dims(X,1)

    enc = LabelEncoder()
    y = enc.fit_transform(y.ravel()).ravel()

    return X, y

print(ds_name)
n_splits=5
n_epoch = 50
batch_size = 32
train_X, train_y = read_mat(ds_name, 'S', 'y', standardize=True)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

index_list = [idx for idx in skf.split(train_X,train_y)]
print(len(index_list), len(index_list[0][0]))
model_list = []
for i, (train_index, _) in enumerate(index_list):
    X_train, y_train = train_X[train_index], train_y[train_index]

    X_train, y_train = preprocess_data(X_train, y_train)

    ds_train = MyDataset(X_train, y_train)
    DL_train = DataLoader(ds_train, batch_size=batch_size)

    model = get_model()
    #model = copy.deepcopy(base_model)
    my_train(model, DL_train, n_epoch=n_epoch, lr=1e-3)
    torch.save(model, "model"+str(i))
    model_list.append(model)

def get_results(fname, user_matrix):

  test_X, test_y = read_mat(fname, user_matrix, 'y', standardize=True)

  acc_list = [] # Metrics for each fold
  f1_list = []
  prec_list = []
  recall_list = []
  d_list = []
  Y = {}

  for i, (_, test_index) in enumerate(index_list):

    # Testing
    X_test, y_test = test_X[test_index], test_y[test_index]
    X_test, y_test = preprocess_data(X_test, y_test)

    model = model_list[i]
    #model = torch.load("model"+str(i))

    y_pred = evaluate(X_test, y_test, model)
    d = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print(d['accuracy'])

    d_list.append(d)
    acc_list.append(d['accuracy'])
    f1_list.append(d['macro avg']['f1-score'])
    prec_list.append(d['macro avg']['precision'])
    recall_list.append(d['macro avg']['recall'])
    Y['pred'] = y_pred
    Y['true'] = y_test

  # Aggregate values over folds
  means = {}
  means['accuracy'] = np.mean(acc_list)
  means['f1'] = np.mean(f1_list)
  means['precision'] = np.mean(prec_list)
  means['recall'] = np.mean(recall_list)

  stds = {}
  stds['accuracy'] = np.std(acc_list)
  stds['f1'] = np.std(f1_list)
  stds['precision'] = np.std(prec_list)
  stds['recall'] = np.std(recall_list)

  return d_list, means, stds, Y

fnames = ['frequency_fix_cr030.mat',
          'frequency_fix_cr050.mat',
          'frequency_fix_cr065.mat',
          'frequency_stat_cr030.mat',
          'frequency_stat_cr050.mat',
          'frequency_stat_cr065.mat',
          'time_peak_cr030.mat',
          'time_peak_cr050.mat',
          'time_peak_cr065.mat']
user_types = ['SA', 'SB', 'S']

import pandas as pd

df_fname = [] 
df_user = []
df_accuracy = []
df_f1 = []
df_precision = []
df_recall = []

archive={}

for fname in fnames:
  archive[fname] = {}
  for user in user_types:
    
    d_list, means, stds, Y = get_results(fname, user)
    print(fname, user)
    print(means['accuracy'])  
    value = stds


    df_fname.append(fname)
    df_user.append(user)
    df_accuracy.append(value['accuracy'])
    df_f1.append(value['f1'])
    df_precision.append(value['precision'])
    df_recall.append(value['recall'])

    archive[fname][user] = (d_list, means, stds, Y)

df_dir = {'fname': df_fname,
          'user': df_user,
          'accuracy': df_accuracy,
          'f1': df_f1,
          'precision': df_precision,
          'recall': df_recall}

df = pd.DataFrame.from_dict(df_dir)
#print(df)
print(df.loc[df['user']=='SB'])

with pd.ExcelWriter('results.xlsx', engine='openpyxl', mode='w') as writer:
    df.loc[df['user']=='SA'].to_excel(writer, sheet_name='UserA')
    df.loc[df['user']=='SB'].to_excel(writer, sheet_name='UserB')
    df.loc[df['user']=='S'].to_excel(writer, sheet_name='Reference')
    writer.save()

writer.close()

d_list, means, stds, Y = get_results(fname, 'X')
archive['reference'] = (d_list, means, stds)

print("Reference")
print('Mean accuracy: ', means['accuracy'])
print('Mean precision: ',means['precision'])
print('Mean recall: ',means['recall'])
print('Mean f1: ',means['f1'])

print('std accuracy: ', stds['accuracy'])
print('std precision: ', stds['precision'])
print('std recall: ', stds['recall'])
print('std f1: ', stds['f1'])

np.save('results',archive)

"""Existing data"""

# From existing results file
import numpy as np
results = np.load('results.npy', allow_pickle=True).item()
print(results.keys())

def get_results(fname, user):
  return results[fname][user]
