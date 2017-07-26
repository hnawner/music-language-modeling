#!/usr/bin/env python

from __future__ import division, print_function
import os, sys
import numpy as np
from math import log
from sklearn.model_selection import KFold
import crnn_utils

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



def neg_log_prob(mels, p_bigrams):

    P = []
    ignoredavg = []
    ignoredtotal = 0

    for mel in mels:
        prev = -1
        ignored = 0
        for note in mel:
            if prev in p_bigrams and note in p_bigrams[prev]:
                P.append(-1*log((p_bigrams[prev])[note]))
            elif prev != -1: ignored += 1
	    
            prev = note

        ignoredavg.append(ignored)
        ignoredtotal += ignored

    mean = np.mean(P)

    print("Negative log probability: ", mean)
    print("Total notes ignored: ", ignoredtotal)
    print("Avg notes ignored: ", np.mean(ignoredavg), "\n")
    return mean


def predict(mels, p_bigrams):

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
    print("Accuracy: ", accuracy, "\n")

    return accuracy


def cv_test(mels):
    accuracy = []
    neglogprob_means = []
    splits = 10 # amount of tests run
    kf = KFold(n_splits=splits, shuffle=True)
    count = 1
    mels = np.asarray(mels)
    
    for train_index, test_index in kf.split(mels):
        train_data, test_data = mels[train_index], mels[test_index]
        p_distr = distribution(train_data)
        print("Test ", count)


        nlp = neg_log_prob(test_data,  p_distr)
        neglogprob_means.append(nlp)

        acc = predict(test_data, p_distr) # returns accuracy
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
    
    print("_______Bigram_______")

    print("___Time/Steps___")
    cv_test(one_h_seqs)

    print("Done.")
    return 0

if __name__ == '__main__':
    main()

