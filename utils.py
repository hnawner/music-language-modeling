#!/usr/bin/env python

import os
import numpy as np

def read_files(folder):
    files = os.listdir(folder)

    maj_mels = []
    min_mels = []

    offsets = { "C":0, "C-sharp":1, "D-flat":1, "D":2, "D-sharp":3, "E-flat":3,
                "E":4, "E-sharp":5, "F-flat":4, "F":5, "F-sharp":6, "G-flat":6,
                "G":7, "G-sharp":8, "A-flat":8, "A":9,
                "A-sharp":10, "B-flat":10, "B":11, "B-sharp":0, "C-flat":11 }

    for f in files:
        path = folder + "/" + f
        offset = 0 # offset from key of C
        is_major = True # default
        with open(path, 'r', 0) as f:
            mel = []
            for line in f:
                parsed = line.split() # delimiter as spaces

                if parsed[0] == "Info" and parsed[1] == "key":
                    if parsed[3] == "Minor": is_major = False
                    offset = offsets[parsed[2]]

                elif parsed[0] == "Note":
                    pitch = int(parsed[3]) - offset
                    mel.append(pitch)

            if is_major: maj_mels.append(mel)
            else: min_mels.append(mel)

    maj_mels_np = np.asarray(maj_mels)
    min_mels_np = np.asarray(min_mels)
    
    return maj_mels_np, min_mels_np


def make_ngrams(seqs, n):
    grams = []
    for seq in seqs:
        prevs = seq[:(n-1)]
        for index in range((n-1), len(seq)):
            prevs += [element]
            grams.append(prevs)
            prevs = prevs[1:]
    return grams


def one_hot_ngram(grams):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []
        for index in range(len(gram) - 1):
            vec = [0] * 88
            vec[(gram[index])] = 1
            vecs.append(vec)
        target = [0] * 88
        target[(gram[-1])] = 1
        :ecs_list.append(vecs)
        targets.append(target)

    return vecs_list, targets


        
def setup(folder, n, mode):
    major, minor = read_files(folder)
    if(mode == "major"):
    	maj_grams = make_ngrams(major, n)
	major_X, major_y = one_hot_ngram(maj_grams)
	major_X_train, major_X_test, major_y_train, major_y_test = tts(major_X, major_y, test_size = 0.2)
	return major_X_train, major_X_test, major_y_train, major_y_test
    else:
	min_grams = make_ngrams(minor, n)
        minor_X, minor_y = one_hot_ngram(min_grams)
	minor_X_train, minor_X_test, minor_y_train, minor_y_test = tts(minor_X, minor_y, test_size = 0.2)
        return minor_X_train, minor_X_test, minor_y_train, minor_y_test
