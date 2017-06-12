#!/usr/bin/env python2.6

#Bigram model with prediction error

#imports:
from __future__ import division
import os
import math
import sys
from read_files import read
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



def cv_test(mels):
    errors = list()
    uk = list()
    mnlps = list()
    splits = 10
    kf = KFold(n_splits = 10, shuffle = True)       
    for train_index, test_index in kf.split(mels):
        train_data, test_data = mels[train_index], mels[test_index]
        int_distr, s_distr = get_distr(train_data)
        err, unkown = predict(int_distr, s_distr, test_data)
        mnlp = mean_neg_log_prob(int_distr, s_distr, test_data)
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
def get_distr(mels):
    intervals = {}
    num_inter = 0
    starts = {}
    num_mels = len(mels)
    

    for mel in mels:
        prev = -1        
        for pitch in mel:
            if(prev == -1):
                if(pitch in starts):
                    starts[pitch] += 1
                else:
                    starts[pitch] = 1
            else:
                interval = pitch - prev
                num_inter += 1
                if(interval in intervals):
                    intervals[interval] += 1
                else:
                    intervals[interval] = 1
            prev = pitch
          
    p_intervals = {}
    p_starts = {}

    for interval in intervals:
        p_intervals[interval] = intervals[interval] / num_inter
    for st in starts:
        p_starts[st] = starts[st] / num_mels 
    
    return p_intervals, p_starts
  

#predicts next note in melody, excluding first note
#prints the percent it guesses correctly    
def predict(p_int, p_st, test):

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

            #start note in melody
            if(prev == -1):
                pred = keywithmaxval(p_st)
            #all other notes
            else:
                interval = keywithmaxval(p_int)
                pred = prev + interval
                
            prev = pitch

            #compare prediction with reality
            if(pred == pitch):
                correct += 1
            
            predictions += 1

    accuracy = correct / predictions

    return (1 - accuracy), unknown

def mean_neg_log_prob(p_int, p_st, test):
    mel_probs = list()

    for mel in test:
        prev = -1
        prob = 0

        for pitch in mel:
            if(prev == -1):
                if(pitch in p_st):
                    prob -= math.log(p_st[pitch])
            else:
                interval = pitch - prev
                if(interval in p_int):
                    prob -= math.log(p_int[interval])
            prev = pitch

        prob = prob / len(mel)
        mel_probs.append(prob)

    mean = np.mean(mel_probs)

    return mean
    

#takes name of melody files 
#directory as user argument
main(sys.argv[1])
