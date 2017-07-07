#!/usr/bin/env python2.6

#Bigram model with prediction error

#imports:
from __future__ import division, print_function
import os
import math
import sys
from utils import read_files as read
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

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

            if(prev_prev != -1 and prev != -1):
                
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

    return p_int_bigrams


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
    ignored = 0
    
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
		            ignored += 1
                
                predictions += 1
                
                #compare prediction with reality
                if(prediction == pitch):
                    correct += 1

            prev_prev = prev
            prev = pitch
            

    #calculate and print percent correct
    accuracy = correct / predictions
    
    print("(ignored) ", ignored)
    print("Total predictions: ", predictions)
    print("Total correct predictions: ", correct)
    print("Accuracy: ", accuracy, "\n")
    
    return accuracy

def neg_log_prob(test, p_int_bg):
    
    P = []
    ignoredavg = []
    ignoredtotal = 0

    for mel in test:
        prev_prev = -1
        prev = -1
        prob = 0
        ignored = 0

        for pitch in mel:

            prev_interval = prev_prev - prev
            interval = pitch - prev

            if(prev_prev != -1 and prev != -1):
                if( (prev_interval in p_int_bg) and (interval in p_int_bg[prev_interval]) ):
                    P.append(-1*math.log((p_int_bg[prev_interval])[interval]))
                elif prev != -1: ignored += 1
                
            ignoredavg.append(ignored)
            ignoredtotal += ignored
                
            prev_prev = pitch
            prev = pitch

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
        int_bg_distr = distribution(train_data)
        print("Test ", count)
        
        acc = predict(test_data, int_bg_distr)
        nlp = neg_log_prob(test_data, int_bg_distr)
        
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

    print("_______Interval Bigram_______")
    
    print("___Major___")
    cv_test(maj_mels)
    print()

    print("___Minor___")
    cv_test(min_mels)
    
    print("Done")
    return 0
    

if __name__ == '__main__':
    main()
