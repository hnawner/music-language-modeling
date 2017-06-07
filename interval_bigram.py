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
        int_bg_distr, int_distr, s_distr = get_distr(train_data)
        err, unknown = predict(int_bg_distr, int_distr, s_distr, test_data)
        mnlp = mean_neg_log_prob(int_bg_distr, int_distr, s_distr, test_data)
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

    
        
    
    


def get_distr(mels):
    int_bigrams = {}
    int_unigrams = {}
    starts = {}
    num_mels = len(mels)
    num_intervals = 0
    

    for mel in mels:
        prev_prev = -1
        prev = -1        
        for pitch in mel:
            prev_interval = prev - prev_prev
            interval = pitch - prev
            if(prev_prev == -1): #start note
                if(pitch in starts):
                    starts[pitch] += 1
                else:
                    starts[pitch] = 1
            
            elif(prev == -1): #second note
                num_intervals += 1
                if(interval in int_unigrams):
                    int_unigrams[interval] += 1
                else:
                    int_unigrams[interval] = 1

            else: #all other notes
                
                num_intervals += 1
                #interval_unigram counts
                if(interval in int_unigrams):
                    int_unigrams[interval] += 1
                else:
                    int_unigrams[interval] = 1

                #interval_bigram counts
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
    p_int_unigrams = {}
    p_starts = {}

    for prev_interval in int_bigrams: #interval bigram probabilities
        p_int_bigrams[prev_interval] = {}
        for interval in int_bigrams[prev_interval]:
            (p_int_bigrams[prev_interval])[interval] = (int_bigrams[prev_interval])[interval] / int_unigrams[prev_interval]

    for interval in int_unigrams: # interval unigram probabilities
        p_int_unigrams[interval] = int_unigrams[interval] / num_intervals
    
    for st in starts:
        p_starts[st] = starts[st] / num_mels 

    return p_int_bigrams, p_int_unigrams, p_starts
  

#predicts next note in melody, excluding first note
#prints the percent it guesses correctly    
def predict(p_int_bg, p_int_ug, p_st, test):

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

            if(prev_prev == -1): #starting note
                prediction = keywithmaxval(p_st)

            elif(prev == -1): #second note
                prediction = prev + keywithmaxval(p_int_ug)

            else: #all other notes
                if(prev_interval in p_int_bg):
                	pred = prev + keywithmaxval(p_int_bg[prev_interval])
                else:
		            unknown += 1
                
            #compare prediction with reality
            if(prediction == pitch):
                correct += 1

            prev_prev = prev
            prev = pitch
            
            predictions += 1

    #calculate and print percent correct
    accuracy = correct / predictions
    return (1 - accuracy), unknown

def mean_neg_log_prob(p_int_bg, p_int_ug, p_st, test):
    mel_probs = list()

    for mel in test:
        prev_prev = -1
        prev = -1
        prob = 0

        for pitch in mel:

            prev_interval = prev_prev - prev
            interval = pitch - prev

            if(prev_prev == -1): #starting note
                if(pitch in p_st):
                    prob -= math.log(p_st[pitch])

            elif(prev == -1): #second note
                if(interval in p_int_ug):
                    prob -= math.log(p_int_ug[interval])
                
            else: #all other notes
                if( (prev_interval in p_int_bg) and (interval in p_int_bg[prev_interval]) ):
                    prob -= math.log((p_int_bg[prev_interval])[interval])

            prev_prev = pitch
            prev = pitch

        prob = prob / len(mel)
        mel_probs.append(prob)

    mean = np.mean(mel_probs)

    return mean
    

#takes name of melody files 
#directory as user argument
main(sys.argv[1])
