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
    print("__________Interval Unigram__________")
    
    print("Major test:")
    cv_test(maj_mel)
    print()

    print("Minor test:")
    cv_test(min_mel)


def dict_softmax(d):
    expD = {np.exp(v) for v in d.values()}
    s = sum(expD)
    softmax = {k: (np.exp(v) / s) for k, v in d.items()}
    return softmax


def cv_test(mels):
    errors = list()
    uk = list()
    mnlps = list()
    splits = 10
    kf = KFold(n_splits = 10, shuffle = True)       
    for train_index, test_index in kf.split(mels):
        train_data, test_data = mels[train_index], mels[test_index]
        int_distr = get_distr(train_data)
        err, unkown = predict(int_distr, test_data)
        mnlp = neg_log_prob(int_distr, test_data)
	uk.append(unkown)
        errors.append(err)
        mnlps.append(mnlp)

    print("  Errors: " + str(errors))
    print("  Mean error: " + str(np.mean(errors)))
    print("  SD errors: " + str(np.std(errors)))
    print("  Mean negative log probabilities: " + str(mnlps))
    print("  Mean of mean negative log probabilities: " + str(np.mean(mnlps)))
    print("  SD of mean negative log probabilities:: " + str(np.std(mnlps)))
    print("  Mean unkown: " + str(np.mean(uk)))


#inputs melodies, splits test/train, 
#calculates start note probabilities
#and conditional probabilities
def distribution(mels):
    intervals = {}
    num_inter = 0
    num_mels = len(mels)
    

    for mel in mels:
        prev = -1        
        for pitch in mel:

            if(prev != -1):
                interval = pitch - prev
                num_inter += 1
                if(interval in intervals):
                    intervals[interval] += 1
                else:
                    intervals[interval] = 1
            prev = pitch
          
    p_intervals = {}


    for interval in intervals:
        p_intervals[interval] = intervals[interval] / num_inter 
    
    #p_intervals = dict_softmax(p_intervals)

    return p_intervals

#predicts next note in melody, excluding first note
#prints the percent it guesses correctly    
def predict(test, p_int):

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

        prev = -1
        
        #loop through notes, excluding first
        for pitch in mel:

            pred = -1

            if(prev != -1):
                interval = keywithmaxval(p_int)
                pred = prev + interval
                predictions += 1
            

                #compare prediction with reality
            	if(pred == pitch):
                    correct += 1

            prev = pitch
            
            

    accuracy = correct / predictions
    print("Accuracy: ", accuracy)

    return (1 - accuracy), unknown

def neg_log_prob(test, p_int):
    P = []

    for mel in test:
        prev = -1
        prob = 0

        for pitch in mel:

            if(prev != -1):
                interval = pitch - prev
                if(interval in p_int):
                    P.append(-1*math.log(p_int[interval]))
            prev = pitch

        ##prob = prob / (len(mel) - 1)
        #mel_probs.append(prob)

    mean = np.mean(P)
    print("Negative log probability: ", mean)
    return mean
    

#takes name of melody files 
#directory as user argument
#main(sys.argv[1])
