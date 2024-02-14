import pandas as pd
import os
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from tensorflow import keras
import numpy as np


def prepare_data(fname):
    # reads data from data (input) file
    data = pd.read_csv(fname)
    # now reads events (output) file
    events_fname = fname.replace('_data', '_events')
    labels = pd.read_csv(events_fname)
    clean = data.drop(['id'], axis=1)  # remove id
    labels = labels.drop(['id'], axis=1)  # remove id
    return clean, labels


auc_tot = []
pred_tot = []
y_tot = []
for subject in range(1, 13):
    y_raw = []
    raw = []
    sequence = []
    for ser in range(2, 9):
        fname = '/Users/Vani/PycharmProjects/bci/grasp_lift_eeg/train/subj'+str(subject)+'_series'+str(ser)+'_data.csv'
        data, labels = prepare_data(fname)
        raw.append(data)
        y_raw.append(labels)
        sequence.extend([ser] * len(data))

print("done reading all data!")
X = pd.concat(raw)
y = pd.concat(y_raw)
# transforms data into numpy arrays
X = np.asarray(X.astype(float))
y = np.asarray(y.astype(int))
sequence = np.asarray(sequence)
print("done converting to numpy arrays!")

cv = model_selection.LeaveOneGroupOut()
cv.get_n_splits(X, y, sequence)
for train, test in cv.split(X, y, sequence):
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # (1302122, 32) (154678, 32) (1302122, 6) (154678, 6)
X_train = np.expand_dims(X_train, axis=2) # reshapes X_train
print(X_train.shape)  # (1302122, 32, 1)
print("ready to train now!")


def evaluate_model(xtrain, ytrain, xtest, ytest):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(32,1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())  # similar to maxpooling1D
    model.add(Dropout(0.5))  # causes nothing to be updated during the backward pass of backpropagation
    model.add(Dense(6, activation='sigmoid'))  # changes dimensions of vector as well
    print("done creating model!")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # this is the loss function which prevents a run-time error
    print("done compiling model!")
    model.summary()

    model.fit(xtrain, ytrain, epochs=10)
    print("about to find accuracy :o ")
    score = model.evaluate(xtest, ytest)
    return score

print(evaluate_model(X_train, y_train, X_test, y_test))
