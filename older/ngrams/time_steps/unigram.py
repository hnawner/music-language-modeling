#!/usr/bin/env python

from __future__ import division, print_function
import os, sys
import numpy as np
from math import log
from sklearn.model_selection import KFold
import crnn_utils

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
    mels = np.asarray(mels)

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
    
    
def one_hot(seqs):
	encoding_dict = {"000":1, "001":2, "010":3, "011":4, "100":5, "101":6, "111":7, "110":8}
	encoded = []
	for seq in seqs:
		encoded_seq = []
		for note in seq:
			note_string = str(note[0]) + str(note[1]) + str(note[2])
			e = encoding_dict[note_string]
			encoded_seq.append(e)
		encoded.append(encoded_seq)
	return encoded
        

def main():
    if len(sys.argv) != 2:
        print("Usage: folder containing mel files")
        return 1
        
    time_step = 125
        
    seqs, files = crnn_utils.read(sys.argv[1])
    encoded = crnn_utils.encode_target(seqs, files, time_step)
    one_h_seqs = one_hot(encoded)
    	
    
    print("_______Unigram_______")

    print("___Time_Steps___")
    cv_test(one_h_seqs)
    

    print("Done.")
    return 0

if __name__ == '__main__':
    main()

