#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from os import listdir
from copy import deepcopy
from sklearn.model_selection import train_test_split

def read(directory):
    seqs = []
    for fi in listdir(directory):
        if fi.endswith(".txt"):
            path = directory + '/' + fi
            with open(path) as f:
                seq = []
                prev_offset = 0
                for line in f:
                    line = line.split()

                    if line[0] == "Note":
                        onset = int(float(line[1]))
                        offset = int(float(line[2]))
                        # triplet edge case:
                        # replace triplets with first triplet, spanning
                        # entire triplets' duration
                        if onset % 10 not in {0, 5} or offset % 10 not in {0, 5}:
                            # check if first occurence of triplets
                            if (seq == [] or
                            isinstance(seq[-1], tuple) and len(seq[-1]) == 2):
                                seq.append(onset)
                                continue
                            continue
                        # check if first occurence after triplets
                        elif len(seq) > 0 and isinstance(seq[-1], int):
                            triplet_onset = seq.pop()
                            seq.append((triplet_onset, prev_offset))
                        # disregard overlapping notes
                        elif offset <= prev_offset: continue
                        # disregard overlapping portion of notes
                        elif onset < prev_offset: onset = prev_offset
                        prev_offset = offset
                        seq.append((onset, offset))
                if isinstance(seq[-1], int): # triplets are end of seq
                    triplet_onset = seq.pop()
                    seq.append((triplet_onset, prev_offset))
                seqs.append(seq)
    return seqs


def encode_input(seqs, time_step):
    # INDICES: LEN 2
    # 0: note is playing
    # 1: note is articulated
    encoded = []
    for s in seqs:
        new = []
        end = s[-1][1]
        for t in range(0, end, time_step):
            if t < s[0][0]:
                new.append([0,0])
            elif t == s[0][0]:
                new.append([1,1])
            elif t > s[0][0] and t < s[0][1]:
                new.append([1,0])
            elif t >= s[0][1] and t < s[1][0]:
                new.append([0,0])
                s.pop(0)
            elif t == s[1][0]:
                new.append([1,1])
                s.pop(0)
            else:
                print('inputs')
                print('time step', t)
                print('s[0]', s[0])
                print('s[1]', s[1])
                raise Exception("nonexhaustive match failure")
        encoded.append(new)
    return encoded


def encode_target(seqs, time_step):
    # INDICES: LEN 3
    # 0: note is articulated
    # 1: note is playing
    # 2: note is not playing
    encoded = []
    for s in seqs:
        new = []
        end = s[-1][1]
        for t in range(0, end, time_step):
            if t < s[0][0]:
                new.append([0,0,1])
            elif t == s[0][0]:
                new.append([1,0,0])
            elif t > s[0][0] and t < s[0][1]:
                new.append([0,1,0])
            elif t >= s[0][1] and t < s[1][0]:
                new.append([0,0,1])
                s.pop(0)
            elif t == s[1][0]:
                new.append([1,0,0])
                s.pop(0)
            else:
                print('targets')
                print('time step', t)
                print('s[0]', s[0])
                print('s[1]', s[1])
                raise Exception("nonexhaustive match failure")
        encoded.append(new)
    return encoded


def pad(X, Y):
    max_len = len(max(X, key=len))
    for i in range(len(X)):
        diff = max_len - len(X[i])
        # destructively modifies X, Y
        X[i] += [[-1,-1]] * diff
        Y[i] += [[0,0,0]] * diff
    return max_len


def make_ngrams(X, n):
    grams = list()
    for L in X:
        prevs = L[:(n-1)]
        for i in range((n-1), len(L)):
            prevs += [ L[i] ]
            grams.append(prevs)
            prevs = prevs[1:]
    return grams


def get_lengths(L):
    lens = deepcopy(L)
    return map(lambda l: len(l), L)


def get_data(directory, time_step):
    seqs = read(directory)
    X = encode_input(deepcopy(seqs), time_step)
    Y = encode_target(seqs, time_step)
    for i in range(len(X)):
        X[i] = X[i][:-1]
        Y[i] = Y[i][8:]
    max_len = pad(X, Y)
    #X = np.asarray(make_ngrams(X, kern_size)).flatten()
    X = np.asarray(X)
    Y = np.asarray(Y)
    split = train_test_split(X, Y, test_size=0.2)
    return split, max_len

