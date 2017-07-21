#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split as tts

def read_files(folder):
    files = os.listdir(folder)

    mels = []

    offsets = { "C":0, "C-sharp":1, "D-flat":1, "D":2, "D-sharp":3, "E-flat":3,
                "E":4, "E-sharp":5, "F-flat":4, "F":5, "F-sharp":6, "G-flat":6,
                "G":7, "G-sharp":8, "A-flat":8, "A":9,
                "A-sharp":10, "B-flat":10, "B":11, "B-sharp":0, "C-flat":11 }

    for f in files:
        path = folder + "/" + f
        offset = 0 # offset from key of C
        with open(path, 'r', 0) as f:
            mel = []
            for line in f:
                parsed = line.split() # delimiter as spaces

                if parsed[0] == "Info" and parsed[1] == "key":
                    offset = offsets[parsed[2]]

                elif parsed[0] == "Note":
                    pitch = int(parsed[3]) - offset
                    mel.append(pitch)

            mels.append(mel)
    
    return mels


def setup_rnn(folder):
    mels = read_files(folder)
    max_len = max_length(mels)
    X, y = make_rnn_data(mels)
    X_pad = pad(X)
    y_pad = pad(y)
    X = rnn_encoder(X_pad)
    y = rnn_labels_encoder(y_pad)
    X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test

def max_length(mels):
    max_len = 0
    for mel in mels:
        if len(mel) > max_len:
            max_len = len(mel)

    return max_len

'''
def pad(mels):
    max_len = max_length(mels)

    for i in range(len(mels)):
        diff = max_len - len(mels[i])
        zeros = [0] * diff
        mels[i] = mels[i] + zeros

    return mels
'''

def pad(X, Y):
    max_len = len(max(X, key=len))
    for i in range(len(X)):
        diff = max_len - len(X[i])
        # destructively modifies X, Y
        X[i] += [[0] * 21] * diff
        Y[i] += [[0] * 95] * diff
    return max_len

def get_mel_lengths(mels):
    lengths = [len(mel) for mel in mels]
    return lengths

def make_rnn_data(mels):
    X = []
    y = []
    for mel in mels:
        X.append(mel[:(len(mel) - 1)])
        y.append(mel[1:])

    return X, y


def rnn_encoder(X):
    X_list = []
    for inst in X:
        start_token = [1] + ([0] * 20)
        vecs = [start_token]
        for pitch in inst:
            start = [0]
            pc_vec = [0] * 12
            octave_vec = [0] * 8
            pc_vec[(pitch % 12)] = 1
            octave_vec[int(pitch / 12)] = 1
            total_vec = start + pc_vec + octave_vec
            vecs.append(total_vec)

        X_list.append(vecs)

    return X_list

def rnn_labels_encoder(y):
    y_list = []
    for inst in y:
        mel_vecs = []
        for pitch in inst:
            vector = [0] * 95
            if(pitch != 0):
                vector[int(pitch)] = 1
            mel_vecs.append(vector)


        y_list.append(mel_vecs)

    return y_list
    
   
