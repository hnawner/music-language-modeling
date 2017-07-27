#!/usr/bin/env python

from __future__ import division, print_function
from utils import read_files as read
import os, sys
import numpy as np
from math import log
from sklearn.model_selection import KFold

def distribution(mels):

    unigrams = {}
    bigrams = {}

    for mel in mels:

        prev = -1

        for pitch in mel:
            
            if (prev != -1):
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

    p_bigrams = {}

    for prev in bigrams:
        p_bigrams[prev] = {} # initialize inner dictionary
        for pitch in bigrams[prev]:
            (p_bigrams[prev])[pitch] = (bigrams[prev])[pitch] / unigrams[prev]

    num_mels = len(mels)

    return p_bigrams
    

def neg_log_prob(pitches, rhythms, p_distr, r_distr):

    P = []
    ignoredavg = []
    ignoredtotal = 0

    for pitch_seq, rhy_seq in zip(pitches, rhythms):
        prev_p = - 1
        prev_r = - 1
        ignored = 0
        for pitch, rhy in zip(pitch_seq, rhy_seq):
            combined_prob = 0
            if prev_p in p_distr and pitch in p_distr[prev_p]:
                combined_prob = (p_distr[prev_p])[pitch]
            if prev_r in r_distr and rhy in r_distr[prev_r]:
                combined_prob *= (r_distr[prev_r])[rhy]
                
            if combined_prob != 0:
                P.append(-1*log(combined_prob))
            elif prev_p == -1 or prev_r == -1: ignored += 1
            
            prev_p = pitch
            prev_r = rhy

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
    ignored = 0

    for pitch_seq, rhy_seq in zip(pitches, rhythms):

        prev_p = -1
        prev_r = -1
        for pitch, rhy in zip(pitch_seq, rhy_seq):
            # prediction
            p_pred = -1
            r_pred = -1
            if prev_p in p_distr:
                p_pred = keywithmaxval(p_distr[prev_p])
            if prev_r in r_distr:
                r_pred = keywithmaxval(r_distr[prev_r])
            if p_pred == pitch and r_pred == rhy: correct += 1
            elif p_pred == -1 or r_pred == -1: ignored += 1
            
            prev_p = pitch
            prev_r = rhy

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