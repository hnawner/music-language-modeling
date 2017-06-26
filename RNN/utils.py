#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split as tts

def read_files(folder):
    files = os.listdir(folder)
    max_length = 0

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
            if len(mel) > max_length:
                max_length = len(mel)
                #print(path)
                #print(len(mel))

            if is_major: maj_mels.append(mel)
            else: min_mels.append(mel)
    
    return maj_mels, min_mels

def pad(mels):
    max_len = 0
    for mel in mels:
        if len(mel) > max_len:
            max_len = len(mel)

    for i in range(len(mels)):
        diff = max_len - len(mels[i])
        zeros = [0] * diff
        mels[i] = mels[i] + zeros

    return mels

def get_mel_lengths(mels):
    lengths = [len(mel) for mel in mels]
    return lengths

def make_rnn_data(mels, length):
    X = []
    y = []
    for mel in mels:
        if len(mel) > (length):
            for i in range(0, (len(mel) - length)):
                new_inst = mel[i:(i + length)]
                new_target = mel[(i + 1):(i + length + 1)]
                X.append(new_inst)
                y.append(new_target)

    return X, y

def make_rnn_data_varlen(mels):
    X = []
    y = []
    for mel in mels:
        X.append(mel[:(len(mel) - 1)])
        y.append(mel[1:])

    return X, y


def rnn_encoder(X):
    X_list = []
    for inst in X:
        vecs = []
        for index in range(len(inst)):
            pc_vec = [0] * 12
            octave_vec = [0] * 8
            pc_vec[(inst[index] % 12)] = 1
            octave_vec[(inst[index] / 12)] = 1
            total_vec = pc_vec + octave_vec
            vecs.append(total_vec)

        X_list.append(vecs)

    return X_list

def rnn_labels_encoder(y):
    y_list = []
    for inst in y:
        mel_vecs = []
        for index in range(len(inst)):
            vector = [0] * 88
            vector[(inst[index])] = 1
            mel_vecs.append(vector)


        y_list.append(mel_vecs)

    return y_list
    
   
   
       


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


        
def setup_ngrams(folder, n, mode, encoder):
    major, minor = read_files(folder)
    if(mode == "major"):
        maj_grams = make_ngrams(major, n)
        major_X, major_y = encoder(maj_grams)
        major_X_train, major_X_test, major_y_train, major_y_test = tts(major_X, major_y, test_size = 0.2)
        return major_X_train, major_X_test, major_y_train, major_y_test
    else:
        min_grams = make_ngrams(minor, n)
        minor_X, minor_y = encoder(min_grams)
        minor_X_train, minor_X_test, minor_y_train, minor_y_test = tts(minor_X, minor_y, test_size = 0.2)
        return minor_X_train, minor_X_test, minor_y_train, minor_y_test

