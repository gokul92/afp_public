import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from functools import reduce
from operator import mul
import sys
import importlib
import os
import dnn

def main():

    # Code to disable tensorflow debugging logs
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    tf.reset_default_graph()

    train_test_frac = 0.5

    # Number of epochs
    N_e = 4000
    # Learning rate
    l_r = 0.001
    # Batch size
    b_size = 256
    # Number of epochs for decay of learning rate 
    epoch_decay = 500
    # Number of levels of orderbook
    level = 5
    # Dropout rate for neural network
    dropout = 0.2

    directory = "/home/ubuntu/run/"
    data = pd.read_csv(directory + "ZC4H_20131205_20140304_clean.csv")
    y_data = np.array(pd.read_csv(directory + "y_ZC4H_20131205_20140304_data.txt", header=None))

    # Drop Time Stamp & Unnamed column
    data = data.drop("#TIMESTAMP", axis=1)
    data = data.drop("Unnamed: 0", axis=1)
    if level == 5:
        data = data.drop("MID_PRICE1", axis=1)
    elif level  == 1:
        data = data.drop("MID_PRICE", axis=1)

    # list of columns to stationarize and normalize
    col_names = data.columns

    N_data = data.shape[0]
    N_train = int(train_test_frac * N_data)
    N_test = N_data - N_train

    # Train set
    X_tr = data.iloc[:N_train]
    y_tr = y_data[:N_train]

    # Test set
    X_te = data.iloc[N_train:]
    y_te = y_data[N_train:]

    # Stationarizing and Normalizing Train & Test set columns
    for col in col_names:
      col_mean = X_tr[col].values.mean()
      col_std = X_tr[col].values.std()

      X_tr[col] = (X_tr[col] - col_mean)/col_std
      X_te[col] = (X_te[col] - col_mean)/col_std

      X_tr[col] = X_tr[col].diff()
      X_te[col] = X_te[col].diff()

    # Dropping rows with NaN values
    X_tr = X_tr.dropna(axis=0).reset_index(drop=True)
    X_te = X_te.dropna(axis=0).reset_index(drop=True)

    if X_tr.isnull().values.any() == True:
        print("NaN exists! Check!")
    else:
        print("x_tr clean - no NaN")

    if X_te.isnull().values.any() == True:
        print("NaN exists! Check!")
    else:
        print("x_te clean - no NaN")

    dnn.traindnn(x_train=X_tr, y_train=y_tr, x_validation=X_te, y_validation=y_te, N_epochs=N_e,
             starter_learning_rate=l_r, batch_size=b_size, directory=directory, restart_flag=0,
             start_epoch=0, epoch_decay=epoch_decay, N_classes = 2, dropout_val = dropout)