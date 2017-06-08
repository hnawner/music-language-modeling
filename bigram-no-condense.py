#!/usr/bin/env python

from __future__ import division, print_function
import os, sys
import numpy as np
from math import log
from sklearn.model_selection import KFold
from read_files import read

def distributions(mels):

    unigrams = {}
    starts = {}
    bigrams = {}

    for mel in mels:

        prev = -1

        for pitch in mel:
            
            if prev == -1:
                if pitch in starts: starts[pitch] += 1
                else: starts[pitch] = 1
            
            else:
                # key to outer bigrams dict = prev
                # key to inner bigrams dict = pitch
                # access value as bigrams[prev][pitch]
                if prev in bigrams:
                    if pitch in bigrams[prev]: (bigrams[prev])[pitch] += 1
                    else: (bigrams[prev])[pitch] = 1
                else:
                    bigrams[prev] = {} # initialize value as dictionary
                    (bigrams[prev])[pitch] = 1 # add pitch to dictionary

            if pitch in unigrams: unigrams[pitch] += 1
            else: unigrams[pitch] = 1

            prev = pitch

    p_starts = {}
    p_bigrams = {}

    for prev in bigrams:
        p_bigrams[prev] = {} # initialize inner dictionary
        for pitch in bigrams[prev]:
            (p_bigrams[prev])[pitch] = (bigrams[prev])[pitch] / unigrams[prev]

    num_mels = len(mels)

    for st in starts:
        p_starts[st] = starts[st] / num_mels

    return p_bigrams, p_starts


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def genre_match(mels, p_bigrams, p_start):

    P = []
    ignoredavg = []
    ignoredtotal = 0

    for mel in mels:
        p = 0
        prev = -1
        ignored = 0
        for note in mel:
            if prev == -1 and note in p_start:
                p -= log(p_start[note])
            elif prev in p_bigrams and note in p_bigrams[prev]:
                p -= log((p_bigrams[prev])[note])
            else: ignored += 1
	    prev = note
        p /= len(mel)
	P.append(p)
        ignoredavg.append(ignored)
        ignoredtotal += ignored

    mean = np.mean(P)

    print("Genre match mean: ", mean)
    print("Total notes ignored: ", ignoredtotal)
    print("Avg notes ignored: ", np.mean(ignoredavg), "\n")
    return mean


def predict(mels, p_bigrams, p_start):

    def keywithmaxval(d):
        # creates a list of keys and vals; returns key with max val
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]

    correct = 0
    predictions = 0
    ignored = 0

    for mel in mels:

        # start prediction
        pred = keywithmaxval(p_start)
        if pred == mel[0]: correct += 1
        predictions += 1
        
        for index in range(1, len(mel)):
            prev = mel[index-1]
            pred = -1 # in case we see a bigram that hasn't occured
            
            if prev in p_bigrams:
                pred = keywithmaxval(p_bigrams[prev])
            else: ignored += 1

            if pred == mel[index]: correct += 1
            predictions += 1

    accuracy = correct / predictions

    print("(ignored) ", ignored)
    print("Total predictions: ", predictions)
    print("Total correct predictions: ", correct)
    print("Percentage correct: ", accuracy, "\n")

    return 1 - accuracy # error


def cross_validation(mels):
    errors = []
    gm_means = []
    splits = 10 # amount of tests run
    kf = KFold(n_splits=splits, shuffle=True)
    
    for train_index, test_index in kf.split(mels):
        train_data, test_data = mels[train_index], mels[test_index]
        p_distr, s_distr = distributions(train_data)

        gm = genre_match(test_data,  p_distr, s_distr)
	gm_means.append(gm)

        e = predict(test_data, p_distr, s_distr) # returns error
        errors.append(e)

    mean = np.mean(errors)
    std = np.std(errors)

    print("Mean error: ", mean)
    print("Standard deviation: ", std)
    
    print("Mean genre match: " + str(np.mean(gm_means)))
    print("Standard deviation genre match: " + str(np.std(gm_means)))

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

