#!/usr/bin/env python

from __future__ import division, print_function
import os, sys
import numpy as np
from math import log
from sklearn.model_selection import KFold
from utils import read_files as read

def distribution(mels):

    unigrams = {}
    total_notes = 0

    for mel in mels:

        for pitch in mel:

            if pitch in unigrams: unigrams[pitch] += 1
            else: unigrams[pitch] = 1

            total_notes += 1
    
    P = {}

    for pitch in unigrams:
        P[pitch] = unigrams[pitch] / total_notes

    #P = dict_softmax(P)

    return P

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def dict_softmax(d):
    expD = {np.exp(v) for v in d.values()}
    s = sum(expD)
    softmax = {k: (np.exp(v) / s) for k, v in d.items()}
    return softmax


def neg_log_prob(mels, dists):

    P = []
    ignoredavg = []
    ignoredtotal = 0

    for mel in mels:
        p = 0
        ignored = 0
        for note in mel:
            if note in dists: P.append(-1*log(dists[note]))
            else: ignored += 1

        #P.append(p/len(mel))
        ignoredavg.append(ignored)
        ignoredtotal += ignored

    print("Negative log probability: ", np.mean(P))
    #print("Total notes ignored: ", ignoredtotal)
    #print("Avg notes ignored: ", np.mean(ignoredavg), "\n")


def predict(mels, P):

    def keywithmaxval(d):
        # creates a list of keys and vals; returns key with max val
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]

    correct = 0
    predictions = 0
    ignored = 0

    for mel in mels:

        for index in range(1, len(mel)):
            # prediction
            if mel[index] not in P: ignored += 1
            elif keywithmaxval(P) == mel[index]: correct += 1

            predictions += 1
    
    accuracy = correct / predictions

    #print("(ignored)", ignored)
    #print("Total predictions: ", predictions)
    print("Accuracy: ", accuracy, "\n")

    return 1 - accuracy


def cross_validation(mels):
    errors = []
    splits = 10
    kf = KFold(n_splits=splits, shuffle=True)
    count = 1

    for train_index, test_index in kf.split(mels):
        train, test = mels[train_index], mels[test_index]

        P = distribution(train)
        print("Test ", count)
        neg_log_prob(test, P)
        e = predict(test, P) # returns error
        errors.append(e)
        count += 1

    mean = np.mean(errors)
    std = np.std(errors)

    print("Mean error: ", mean)
    print("Standard deviation: ", std)
    print()

def main():
    if len(sys.argv) != 2:
        print("Usage: folder containing mel files")
        return 1

    maj_mels, min_mels = read(sys.argv[1])

    cross_validation(maj_mels)
    cross_validation(min_mels)

    print("Done.")
    return 0

if __name__ == '__main__':
    main()

