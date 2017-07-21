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

    mels_np = np.asarray(maj_mels)
    
    return mels_np


def one_hot(mels):
    one_hots_main = []

    for mel in mels:
        one_hots = []

        for pitch in mel:
            vec = [0] * 88
            vec[pitch] = 1
            one_hots += vec

        one_hots_main.append(one_hots)

    return one_hots_main


def padding(one_hots, max_len):

    def f(a):
        a += [0] * (88 * (max_len - len(a)))
        return a

    return map(f, one_hots)

def setupRNN(path):
    mels = read_files(path)
    mels = one_hot(mels)

    max_len = 0
    for mel in mels:
        l = len(mel)
        if l > max_len: max_len = l

    X = y = mels
    for i in range(len(mels)):
        X[i] = X[i][:-1]
        y[i] = y[i][1:]

    X = padding(X, max_len)
    y = padding(y, max_len)

    return X, y, max_len
