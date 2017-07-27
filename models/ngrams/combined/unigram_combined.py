#!/usr/bin/env python

from __future__ import division, print_function
from utils import read_files as read
import os, sys
import numpy as np
from math import log
from sklearn.model_selection import KFold

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
    

def neg_log_prob(pitches, rhythms, p_distr, r_distr):

    P = []
    ignoredavg = []
    ignoredtotal = 0

    for pitch_seq, rhy_seq in zip(pitches, rhythms):
        ignored = 0
        for pitch, rhy in zip(pitch_seq, rhy_seq):
            if pitch in p_distr and rhy in r_distr:
                combined_prob = p_distr[pitch] *  r_distr[rhy]
                P.append(-1*log(combined_prob))
            else: ignored += 1

        ignoredavg.append(ignored)
        ignoredtotal += ignored
        
    mean = np.mean(P)

    print("Negative log probability: ", mean)
    print("Total notes ignored: ", ignoredtotal)
    print("Avg notes ignored: ", np.mean(ignoredavg), "\n")
    return mean 
    
    
def predict(pitches, rhythms, p_distr, r_distr):

    def keywithmaxval(d):
        # creates a list of keys and vals; returns key with max val
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]

    correct = 0
    predictions = 0

    for pitch_seq, rhy_seq in zip(pitches, rhythms):

        for pitch, rhy in zip(pitch_seq, rhy_seq):
            # prediction
            p_pred = keywithmaxval(p_distr)
            r_pred = keywithmaxval(r_distr)
            if p_pred == pitch and r_pred == rhy: correct += 1

            predictions += 1
    
    accuracy = correct / predictions

    print("Total predictions: ", predictions)
    print("Total correct predictions: ", correct)
    print("Accuracy: ", accuracy, "\n")

    return accuracy
    
       

def cv_test(pitches, rhythms):
    accuracy = []
    neglogprob_means = []
    splits = 10
    kf = KFold(n_splits=splits, shuffle=True)
    count = 1
    pitches = np.asarray(pitches)
    rhythms = np.asarray(rhythms)


    for train_index, test_index in kf.split(pitches):
        train_p, test_p = pitches[train_index], pitches[test_index]
        train_r, test_r = rhythms[train_index], rhythms[test_index]

        P = distribution(train_p)
        R = distribution(train_r)
        
        print("Test ", count)
        
        nlp = neg_log_prob(test_p, test_r, P, R)
        neglogprob_means.append(nlp)
        
        acc = predict(test_p, test_r, P, R)
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
    train = str(sys.argv[1])
    test = str(sys.argv[2])


    pitches, rhythms = read(train, True)
    
    print("_______Unigram_______")

    print("___Combined___")
    cv_test(pitches, rhythms)

    print("Done.")
    return 0

if __name__ == '__main__':
    main()