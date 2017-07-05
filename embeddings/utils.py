#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split as tts

def read_files(folder):
    files = os.listdir(folder)

    mels = []

    offsets = { "C":0, "C-sharp":1, "D-flat":1, "D":2, "D-sharp":3, "E-flat":3,
                "E":4, "E-sharp":5, "F-flat":4, "F":5, "F-sharp":6, "G-flat":6,
                "G":7, "G-sharp":8, "A-flat":8, "A":9,
                "A-sharp":10, "B-flat":10, "B":11, "B-sharp":0, "C-flat":11 }

    for f in files:
        path = folder + "/" + f
        offset = 0 # offset from key of C
        with open(path, 'r', 0) as f:
            mel = []
            for line in f:
                parsed = line.split() # delimiter as spaces

                if parsed[0] == "Info" and parsed[1] == "key":
                    offset = offsets[parsed[2]]

                elif parsed[0] == "Note":
                    pitch = int(parsed[3]) - offset
                    mel.append(pitch)

            mels.append(mel)
    
    return mels


def make_ngrams(seqs, n):
    grams = []
    for seq in seqs:
        prevs = seq[:(n-1)]
        for index in range((n-1), len(seq)):
            prevs += [ (seq[index]) ]
            grams.append(prevs)
            prevs = prevs[1:]
    return grams


def one_hot_ngram_PCandOctave(grams):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []
        for index in range(len(gram) - 1):
            pc_vec = [0] * 12
            octave_vec = [0] * 8
            pc_vec[(gram[index] % 12)] = 1
            octave_vec[(gram[index] / 12)] = 1
            vecs += pc_vec
            vecs += octave_vec
        target = gram[-1]
        vecs_list.append(vecs)
        targets.append(target)

    return vecs_list, targets


def one_hot_ngram(grams):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []
        for index in range(len(gram) - 1):
            vec = [0] * 88
            vec[(gram[index])] = 1
            vecs += vec
        target = gram[-1]
        vecs_list.append(vecs)
        targets.append(target)

    return vecs_list, targets


def one_hot_ngram_AbsAndPc(grams):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []
        for index in range(len(gram) - 1):
            absvec = [0] * 88
            pcvec = [0] * 12
            absp = gram[index]
            pc = int(absp % 12)
            absvec[absp] = 1
            pcvec[pc] = 1
            absvec += pcvec
            vecs += absvec
        target = gram[-1]
        vecs_list.append(vecs)
        targets.append(target)

    return vecs_list, targets


def one_hot_ngram_CNN(grams, n):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []
        for index in range(len(gram) - 1):
            vec = np.array([0] * 88)
            vec[(gram[index])] = 1
            vecs.append(vec)
        vecs = np.reshape(np.array((vecs), ndmin = 3), [1, 88, (n-1)])
        target = np.array(gram[-1])
        vecs_list.append(vecs)
        targets.append(target)

    return np.array(vecs_list), np.array(targets)

        
def setup_ngrams(folder, n, encoder):
    mels = read_files(folder)
    grams = make_ngrams(mels, n)
    X, y = encoder(grams, n)
    return tts(X, y, test_size = 0.2)


def one_hot(grams):
    vecs_list = []
    targets = []
    for g in grams:
        vecs = []
        for i in range(len(g) - 1):
            vec = np.array([0] * 88)
            vec[(g[i])] = 1
            vecs.append(vec)
        target = g[-1]
        targets.append(target)
        vecs_list.append(vecs)

    return vecs_list, targets


def setupSNN(data, ngramlen):
    data = read_files(data)
    grams = make_ngrams(data, ngramlen)
    X, y = one_hot(grams)
    return tts(X, y, test_size = 0.2)
