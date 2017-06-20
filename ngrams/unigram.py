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

    return P


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

        ignoredavg.append(ignored)
        ignoredtotal += ignored
        
    mean = np.mean(P)

    print("Negative log probability: ", mean)
    print("Total notes ignored: ", ignoredtotal)
    print("Avg notes ignored: ", np.mean(ignoredavg), "\n")
    return mean


def predict(mels, P):

    def keywithmaxval(d):
        # creates a list of keys and vals; returns key with max val
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]

    correct = 0
    predictions = 0

    for mel in mels:

        for index in range(1, len(mel)):
            # prediction
            if keywithmaxval(P) == mel[index]: correct += 1

            predictions += 1
    
    accuracy = correct / predictions

    print("Total predictions: ", predictions)
    print("Total correct predictions: ", correct)
    print("Accuracy: ", accuracy, "\n")

    return accuracy


def cv_test(mels):
    accuracy = []
    neglogprob_means = []
    splits = 10
    kf = KFold(n_splits=splits, shuffle=True)
    count = 1

    for train_index, test_index in kf.split(mels):
        train, test = mels[train_index], mels[test_index]
        P = distribution(train)
        print("Test ", count)
        
        nlp = neg_log_prob(test, P)
        neglogprob_means.append(nlp)
        
        acc = predict(test, P) # returns accuracy
        accuracy.append(acc)
        
        count += 1

    mean = np.mean(accuracy)
    std = np.std(accuracy)

    print("**Overall**")

    print("Mean accuracy: ", mean)
    print("Standard deviation accuracy: ", std)
    
    print("Mean negative log probability: " + str(np.mean(neglogprob_means)))
    print("Standard deviation negative log probability: " + str(np.std(neglogprob_means)))

def main():
    if len(sys.argv) != 2:
        print("Usage: folder containing mel files")
        return 1

    maj_mels, min_mels = read(sys.argv[1])
    
    print("_______Unigram_______")

    print("___Major___")
    cv_test(maj_mels)
    
    print("___Minor___")
    cv_test(min_mels)

    print("Done.")
    return 0

if __name__ == '__main__':
    main()

