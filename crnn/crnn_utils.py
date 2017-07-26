#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from os import listdir
from copy import deepcopy
from sklearn.model_selection import train_test_split

def read(directory, pitch_min, pitch_max):
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
                        if pitch < pitch_min or pitch >= pitch_max:
                            print('''midi pitch outside accepted range;
                                     increasing range to fit.''')
                            if pitch < pitch_min: pitch_min = pitch
                            else: pitch_max = pitch + 1
                        # triplet edge case:
                        # replace triplets with first triplet, spanning
                        # entire triplets' duration
                        if onset % 10 not in {0, 5} or offset % 10 not in {0, 5}:
                            # check if first occurence of triplets
                            if seq == [] or len(seq[-1]) == 3:
                                seq.append((pitch, onset))
                            continue
                        # check if first occurence after triplets
                        elif seq != [] and len(seq[-1]) == 2:
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
    return rests_as_notes(seqs), pitch_min, pitch_max


def rests_as_notes(seqs):
    for s in seqs:
        new = list()
        for i in range(len(s)-1):
            new.append(s[i])
            # if there is a rest between 2 notes
            if s[i+1][0] - s[i][1] > 0:
                onset = s[i][1]
                offset = s[i+1][0]
                new.append((onset, offset, -1)) # denotes rest
        s = new
    return seqs


def encode(seqs, time_step, pitch_range):
    # INDICES: LEN (2 + pitch_range + 1)
    # based off Magenta project encoder/decoder
    # 0: no event
    # 1: note-off event (start of rest)
    # 2 -> pitch_range: note-on event and pitch of note
    #                   where 2 == min(pitch_range),
    #                   2 + pitch_range - 1 == max(pitch_range)
    # len - 1: start token
    encoded = []
    pr = pitch_range[1] - pitch_range[0]
    for s in seqs:
        new = []
        # need 16 start tokens
        for i in range(16):
            vec = [0] * (3 + pr)
            vec[-1] = 1
            new.append(vec)
        t = 0
        for onset, offset, pitch in s:
            while t < offset:
                one_hot = [0] * (3 + pr)
                if t < onset:
                    one_hot[0] = 1
                elif t == onset:
                    index = 1 if onset == -1 else (pitch - pr + 2)
                    one_hot[index] = 1
                else:
                    one_hot[0] = 1
                new.append(one_hot)
                t += time_step
        encoded.append(new)
    return encoded


def pad(X, Y, pitch_range):
    max_len = len(max(X, key=len))
    pr = pitch_range[1] - pitch_range[0]
    for i in range(len(X)):
        empty = [0] * (3 + pr)
        diff = max_len - len(X[i])
        # destructively modifies X, Y
        X[i] += [empty] * diff
        Y[i] += [empty] * diff
    return max_len


# REQUIRES: pitch_range has form [min, max)
def get_data(directory, time_step, pitch_min, pitch_max):
    seqs, pitch_min, pitch_max = read(directory, pitch_min, pitch_max)
    X = encode(seqs, time_step, (pitch_min, pitch_max))
    Y = deepcopy(X)
    for i in range(len(X)):
        X[i] = X[i][:-1] # doesn't give last note
        Y[i] = Y[i][16:] # removes start tokens
    max_len = pad(X, Y, (pitch_min, pitch_max))
    X = np.asarray(X)
    Y = np.asarray(Y)
    split = train_test_split(X, Y, test_size=0.2)
    return split, max_len

