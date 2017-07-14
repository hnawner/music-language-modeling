#!/usr/bin/env python

from __future__ import division, print_function
import os, sys
import numpy as np
from math import log
from sklearn.model_selection import KFold
import utils

important = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def distribution(mels, keys): 

    maj_pc_counts = [0] * 12
    min_pc_counts = [0] * 12
    maj_total_notes = 0
    min_total_notes = 0
    
    transposed_mels = utils.transpose(mels, keys)

    for mel, key in zip(transposed_mels, keys):

        for pitch in mel:
            
            pc = (int(pitch) % 12)
            
            if (pc in important) == False:
            	continue

            if key[1] == 1: 
                maj_pc_counts[pc] += 1
                maj_total_notes += 1
            else: 
                min_pc_counts[pc] += 1
                min_total_notes += 1

    
    maj_distr = [ (p * 1.0) / maj_total_notes for p in maj_pc_counts]
    min_distr = [ (p * 1.0) / min_total_notes for p in min_pc_counts]
    
    #print(maj_distr)
    #print("\n")
    #print(min_distr)
    #quit()


    return maj_distr, min_distr


def predict(test_X, test_y, maj_distr, min_distr):

    def keywithmaxval(d):
        # creates a list of keys and vals; returns key with max val
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]

    correct = 0
    predictions = 0

    for mel, key in zip(test_X, test_y):
        
        mel_counts = [0] * 12
        notes = 0
        for pitch in mel:
            pc = pitch % 12
            mel_counts[pc] += 1
            notes += 1
        mel_distr = np.asarray([ (c * 1.0) / notes for c in mel_counts])
        
        maj_compare = [ np.dot(np.roll(mel_distr, r), np.asarray(maj_distr)) for r in range(12)]
        
        min_compare = [ np.dot(np.roll(mel_distr, r), np.asarray(min_distr)) for r in range(12)]

        maj_max = np.argmax(maj_compare)
        min_max = np.argmax(min_compare)
        
        prediction = [(12 - min_max), 0]
        if maj_compare[maj_max] > min_compare[min_max]:
            prediction = [(12 - maj_max), 1]


        if key[0] == prediction[0] and key[1] == prediction[1]:
            correct += 1

        predictions += 1
    
    accuracy = correct / predictions

    #print("Total predictions: ", predictions)
    #print("Total correct predictions: ", correct)
    #print("Accuracy: ", accuracy, "\n")

    return accuracy


def cv_test(mels, keys, length):
    print("Length ", length)
    accuracy = []
    splits = 5
    kf = KFold(n_splits=splits, shuffle=True)
    count = 1
    mels = np.asarray(mels)
    keys = np.asarray(keys)

    for train_index, test_index in kf.split(mels):
        train_X, test_X = mels[train_index], mels[test_index]
        train_y, test_y = keys[train_index], keys[test_index]
        
        maj_distr, min_distr = distribution(train_X, train_y)
        test_X, test_y = utils.make_id_data(test_X, test_y, length)
        
        #print("Test ", count)
        
        acc = predict(test_X, test_y, maj_distr, min_distr)
        accuracy.append(acc)
        
        count += 1

    mean = np.mean(accuracy)
    std = np.std(accuracy)

    print("**Overall**")

    print("Mean accuracy: ", mean)
    print("Standard deviation accuracy: ", std)

def main():
    if len(sys.argv) != 2:
        print("Usage: folder containing mel files")
        return 1
        
    mels, keys = utils.read_files(sys.argv[1], "key")
    
    
    print("_______Key_ID_______")

    for l in range(1, 25):
        cv_test(mels, keys, l)
    

    print("Done.")
    return 0

if __name__ == '__main__':
    main()

