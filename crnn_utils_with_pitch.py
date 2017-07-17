#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from os import listdir
from copy import deepcopy
from sklearn.model_selection import train_test_split

def read(directory, pitch_range):
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
                        # weird conversion bc adrian sux
                        onset = int(float(line[1]))
                        offset = int(float(line[2]))
                        pitch = int(float(line[3]))
                        if pitch < pitch_range[0] or pitch >= pitch_range[1]:
                            print('midi pitch outside accepted range; skipping.')
                            break
                        # triplet edge case:
                        # replace triplets with first triplet, spanning
                        # entire triplets' duration
                        if onset % 10 not in {0, 5} or offset % 10 not in {0, 5}:
                            # check if first occurence of triplets
                            if seq == [] or len(seq[-1]) == 3:
                                seq.append((pitch, onset))
                            continue
                        # check if first occurence after triplets
                        elif seq == [] or len(seq[-1]) == 2:
                            trip_pitch, trip_onset = seq.pop()
                            seq.append((trip_onset, prev_offset, trip_pitch))
                        # disregard overlapping notes
                        elif offset <= prev_offset:
                            continue
                        # disregard overlapping portion of notes
                        elif onset < prev_offset:
                            onset = prev_offset
                        seq.append((onset, offset, pitch))
                        prev_offset = offset
                # edge case: triplets end song
                if len(seq[-1]) == 2:
                    trip_pitch, trip_onset = seq.pop()
                    seq.append((trip_onset, prev_offset, trip_pitch))
                seqs.append(seq)
    return seqs


def encode(seqs, time_step, pitch_range):
    # INDICES: LEN (2 + pitch_range)
    # based off Magenta project encoder/decoder
    # 0: no event
    # 1: note-off event (start of rest)
    # 2 -> pitch_range: note-on event and pitch of note
    #                   where 2 == min(pitch_range),
    #                   2 + pitch_range - 1 == max(pitch_range)
    encoded = []
    pr = pitch_range[1] - pitch_range[0]
    for s in seqs:
        new = []
        end = s[-1][1]
        for t in range(0, end, time_step):
            one_hot = [0] * (2 + pr)

            if t == s[0][0]:
                index = (s[0][2] - pr) + 2
                one_hot[index] = 1
                new.append(one_hot)
            elif t == s[0][1]:
                one-hot[1] = 1
                new.append(one_hot)
                s.pop(0)
            elif t == s[1][0]:
                index = (s[0][2] - pr) + 2
                one_hot[index] = 1
                new.append(one_hot)
                s.pop(0)
            elif t == s[1][1]:
                one_hot[1] = 1
                new.append(one_hot)
                s.pop(0)
                s.pop(0) # not an accident, will remove what was previously s[1]
            else:
                one_hot[0] = 1
                new.append(one_hot)
        encoded.append(new)
    return encoded


def pad(X, Y, pitch_range):
    max_len = len(max(X, key=len))
    pr = pitch_range[1] - pitch_range[0]
    for i in range(len(X)):
        empty = [0] * (2 + pr)
        diff = max_len - len(X[i])
        # destructively modifies X, Y
        X[i] += [empty] * diff
        Y[i] += [empty] * diff
    return max_len

'''
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
'''

# REQUIRES: pitch_range has form [min, max)
def get_data(directory, time_step, pitch_range):
    seqs = read(directory, pitch_range)
    X = encode(seqs, time_step, pitch_range)
    Y = deepcopy(encoded)
    for i in range(len(X)):
        X[i] = X[i][:-1]
        Y[i] = Y[i][8:]
    max_len = pad(X, Y, pitch_range)
    X = np.asarray(X)
    Y = np.asarray(Y)
    split = train_test_split(X, Y, test_size=0.2)
    return split, max_len

