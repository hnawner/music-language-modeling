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
    has_triplets = False # for now
    for fi in listdir(directory):
        if fi.endswith(".mel"):
            path = directory + '/' + fi
            with open(path) as f:
                seq = []
                prev_offset = 0
                for line in f:
                    line = line.split()

                    if line[0] == "Note":
                        onset = int(line[1])
                        offset = int(line[2])
                        # should get rid of mels with triplets
                        if onset % 10 not in {0, 5} or offset % 10 not in {0, 5}:
                            has_triplets = True
                            break
                        if offset <= prev_offset: continue
                        if onset < prev_offset:
                            onset = prev_offset
                        prev_offset = offset
                        seq.append((onset, offset))
                if has_triplets: continue
                seqs.append(seq)
    return seqs


def encode_input(seqs):
    # INDICES: LEN 2
    # 0: note is playing
    # 1: note is articulated
    TIME_STEP = 125
    encoded = []
    for s in seqs:
        new = []
        end = s[-1][1]
        for t in range(0, end, TIME_STEP):
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


def encode_target(seqs):
    # INDICES: LEN 2
    # 0: note is articulated
    # 1: note is playing
    # 2: note is not playing
    TIME_STEP = 125
    encoded = []
    for s in seqs:
        new = []
        end = s[-1][1]
        for t in range(0, end, TIME_STEP):
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

def get_lengths(L):
    lens = deepcopy(L)
    return map(lambda l: len(l), L)

def get_data(directory):
    seqs = read(directory)
    X = encode_input(seqs)
    Y = encode_target(seqs)
    for i in range(len(X)):
        X[i] = X[i][:-1]
        Y[i] = Y[i][8:]
    max_len = pad(X, Y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    split = train_test_split(X, Y, test_size=0.2)
    return split, max_len

