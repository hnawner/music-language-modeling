#!/usr/bin/env python2.6


from __future__ import division, print_function
import os
import math
import sys
from utils import read_files as read
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np


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
    
    return p_intervals


def predict(test, p_int):

    def keywithmaxval(d):
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]
    
    #predictions counters
    predictions = 0
    correct = 0
    
    for mel in test:

        prev = -1
        
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
    
    print("Total predictions: ", predictions)
    print("Total correct predictions: ", correct)
    print("Accuracy: ", accuracy, "\n")

    return accuracy


def neg_log_prob(test, p_int):
    
    P = []
    ignoredtotal = 0
    ignoredavg = []

    for mel in test:
        prev = -1
        prob = 0
        ignored = 0

        for pitch in mel:

            if(prev != -1):
                interval = pitch - prev
                if(interval in p_int):
                    P.append(-1*math.log(p_int[interval]))
                else: ignored += 1
            
            prev = pitch
            
            ignoredavg.append(ignored)
            ignoredtotal += ignored


    mean = np.mean(P)
    
    print("Negative log probability: ", mean)
    print("Total notes ignored: ", ignoredtotal)
    print("Avg notes ignored: ", np.mean(ignoredavg), "\n")
    return mean
    
def cv_test(mels):
    accuracy = []
    neglogprob_means = []
    splits = 10
    kf = KFold(n_splits = 10, shuffle = True)    
    count = 1
    
    for train_index, test_index in kf.split(mels):
        train_data, test_data = mels[train_index], mels[test_index]
        int_distr = distribution(train_data)
        print("Test ", count)
        
        acc = predict(test_data, int_distr)
        nlp = neg_log_prob(test_data, int_distr)
        
        accuracy.append(acc)
        neglogprob_means.append(nlp)
        
        count += 1
    
    print("**Overall**")

    print("Mean accuracy: ", np.mean(accuracy))
    print("Standard deviation accuracy: ", np.std(accuracy))
    
    print("Mean negative log probability: ", np.mean(neglogprob_means))
    print("Standard deviation negative log probability: ", np.std(neglogprob_means))



    
def main():
    if len(sys.argv) != 2:
        print("Usage: folder containing mel files")
        return 1

    maj_mels, min_mels = read(sys.argv[1])

    print("_______Interval Unigram_______")
    
    print("___Major___")
    cv_test(maj_mels)
    print()

    print("___Minor___")
    cv_test(min_mels)
    
    print("Done")
    return 0

if __name__ == '__main__':
    main()
