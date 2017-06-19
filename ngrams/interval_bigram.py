#!/usr/bin/env python2.6

#Bigram model with prediction error

#imports:
from __future__ import division, print_function
import os
import math
import sys
from utils import read_files
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

#outputs results
def main(*args):
    if(len(args)!=1):
        print("error")

    maj_mel, min_mel = read(args[0])

    print("_______Interval Bigram_______")
    
    print("Major test:")
    cv_test(maj_mel)
    print()

    print("Minor test:")
    cv_test(min_mel)



def cv_test(mels):
    errors = list()
    uk = list()
    mnlps = list()
    splits = 10
    kf = KFold(n_splits = 10, shuffle = True)       
    for train_index, test_index in kf.split(mels):
        train_data, test_data = mels[train_index], mels[test_index]
        int_bg_distr = distribution(train_data)
        err, unknown = predict(int_bg_distr test_data)
        mnlp = neg_log_prob(int_bg_distr, test_data)
        errors.append(err)
        mnlps.append(mnlp)
        uk.append(unknown)

    print("  Errors: " + str(errors))
    print("  Mean error: " + str(np.mean(errors)))
    print("  SD errors: " + str(np.std(errors)))
    print("  Mean negative log probabilities: " + str(mnlps))
    print("  Mean of mean negative log probabilities: " + str(np.mean(mnlps)))
    print("  SD of mean negative log probabilities: " + str(np.std(mnlps)))
    print("  Avg unknown: " + str(np.mean(uk)))



def distribution(mels):
    int_bigrams = {}
    int_unigrams = {}
    num_mels = len(mels)
    num_intervals = 0
    

    for mel in mels:
        prev_prev = -1
        prev = -1        
        for pitch in mel:
            prev_interval = prev - prev_prev
            interval = pitch - prev

            if(prev_prev != -1 and prev != -1): #all other notes
                
                num_intervals += 1
                
                # interval_unigram counts
                if(interval in int_unigrams):
                    int_unigrams[interval] += 1
                else:
                    int_unigrams[interval] = 1

                # interval_bigram counts
                if(prev_interval in int_bigrams):
                    if(interval in int_bigrams[prev_interval]):
                        (int_bigrams[prev_interval])[interval] += 1
                    else:
                        (int_bigrams[prev_interval])[interval] = 1
                else:
                    int_bigrams[prev_interval] = {}
                    (int_bigrams[prev_interval])[interval] = 1

            prev_prev = prev
            prev = pitch
          
    p_int_bigrams = {}

    for prev_interval in int_bigrams: #interval bigram probabilities
        p_int_bigrams[prev_interval] = {}
        for interval in int_bigrams[prev_interval]:
            (p_int_bigrams[prev_interval])[interval] = (int_bigrams[prev_interval])[interval] / int_unigrams[prev_interval]

    #for prev in p_int_bigrams:
    #    p_int_bigrams[prev] = dict_softmax(p_int_bigrams[prev])

    return p_int_bigrams


def dict_softmax(d):
    expD = {np.exp(v) for v in d.values()}
    s = sum(expD)
    softmax = {k: (np.exp(v) / s) for k, v in d.items()}
    return softmax


#predicts next note in melody, excluding first note
#prints the percent it guesses correctly    
def predict(test, p_int_bg):

    def keywithmaxval(d):
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]
    
    #predictions counters
    predictions = 0
    correct = 0
    unknown = 0
    
    #loop through melodies
    for mel in test:
        
        prev_prev = -1
        prev = -1       

        #loop through notes, excluding first
        for pitch in mel:

            prev_interval = prev - prev_prev
            prediction = -1

            if(prev_prev != -1 and prev != -1): #all other notes
                if(prev_interval in p_int_bg):
                	prediction = prev + keywithmaxval(p_int_bg[prev_interval])
                else:
		            unknown += 1
                
                predictions += 1
                
                #compare prediction with reality
                if(prediction == pitch):
                    correct += 1

            prev_prev = prev
            prev = pitch
            

    #calculate and print percent correct
    accuracy = correct / predictions
    print("Accuracy: ", accuracy)
    return (1 - accuracy), unknown

def neg_log_prob(test, p_int_bg):
    P = []

    for mel in test:
        prev_prev = -1
        prev = -1
        prob = 0

        for pitch in mel:

            prev_interval = prev_prev - prev
            interval = pitch - prev

            if(prev_prev != -1 and prev != -1): #all other notes
                if( (prev_interval in p_int_bg) and (interval in p_int_bg[prev_interval]) ):
                    P.append(-1*math.log((p_int_bg[prev_interval])[interval]))

            prev_prev = pitch
            prev = pitch

        #prob = prob / (len(mel) - 2)
        #mel_probs.append(prob)

    mean = np.mean(P)
    print("Negative log probability: ", mean)

    return mean
    

#takes name of melody files 
#directory as user argument
#main(sys.argv[1])
