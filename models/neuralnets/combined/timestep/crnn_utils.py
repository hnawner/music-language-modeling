#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from os import listdir
from copy import deepcopy
from sklearn.model_selection import train_test_split


def read(directory, time_step, pitch_min, pitch_max, trans = True):

    key_offsets = { "C":0, "C-sharp":1, "D-flat":1, "D":2, "D-sharp":3, "E-flat":3,
                "E":4, "E-sharp":5, "F-flat":4, "F":5, "F-sharp":6, "G-flat":6,
                "G":7, "G-sharp":8, "A-flat":8, "A":9,
                "A-sharp":10, "B-flat":10, "B":11, "B-sharp":0, "C-flat":11 }

    ts = time_step # time step
    seqs = []
    files = []
    for fi in listdir(directory):
        if fi.endswith(".txt"):
            path = directory + '/' + fi
            with open(path) as f:
                seq = []
                key_off = 0
                replace_length = 0
                replace_onset = None
                replace_pitch = None
                for line in f:
                    line = line.split()

                    if trans and line[0] == "*K":
                        key_off = key_offsets[str(line[1])]

                    elif line[0] == "Note":
                        onset = int(float(line[1])) 
                        offset = int(float(line[2]))
                        length = offset - onset
                        pitch = int(float(line[3])) - key_off

                        # adjust pitch range if necesary
                        if pitch < pitch_min or pitch >= pitch_max:
                            print('''midi pitch outside accepted range;
                                     increasing range to fit.''')
                            if pitch < pitch_min: pitch_min = pitch
                            else: pitch_max = pitch + 1

                        if seq == [] and replace_length == 0: # first note
                            if onset != 0: # starts with rest
                                seq.append([0, onset, -1])
                        if seq != [] and (seq[-1][1] + replace_length) < onset: #rest
                            rest_length = onset - (seq[-1][1] + replace_length)
                            if rest_length % ts != 0: # non-recognized note value
                                if replace_length == 0:
                                    replace_length = rest_length
                                    replace_pitch = 2
                                    replace_onset = seq[-1][1]
                                else:
                                    replace_length += rest_length
                            else: # acceptable note value
                                if replace_length != 0: 
                                    if replace_length % ts == 0:
                                        seq.append([replace_onset, (replace_onset + replace_length),replace_pitch])

                                        replace_pitch = None
                                        replace_onset = None
                                        replace_length = 0
                                    else:
                                        print("rest HELP ME!!")  
                                        print(fi) 
                                seq.append([seq[-1][1], onset, -1])
                        
                        if length % ts != 0: # non-recognized note value
                            if replace_length == 0:
                                replace_length = length
                                replace_pitch = pitch
                                replace_onset = onset
                            else:
                                replace_length += length
                        else: # acceptable note value
                            if replace_length != 0: 
                                if replace_length % ts == 0:
                                    seq.append([replace_onset, (replace_onset + replace_length), replace_pitch])
                                    replace_pitch = None
                                    replace_onset = None
                                    replace_length = 0
                                else:
                                    print("HELP ME!!")  
                                    print(fi)   
                                    print(line)
                                    print(replace_length)    
                                    print(replace_onset)    
                                    print(replace_pitch)    
                            seq.append([onset, offset, pitch])
                        if replace_length !=0 and replace_length % ts == 0:
                            seq.append([replace_onset, (replace_onset + replace_length), replace_pitch])
                            replace_pitch = None
                            replace_onset = None
                            replace_length = 0
                seqs.append(seq)
                files.append(fi)
    return seqs, pitch_min, pitch_max





'''
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
'''


def encode(seqs, time_step, pitch_range, is_target, start = None):
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
        # X needs 16 start tokens
        if is_target == False:
            for i in range(start):
                vec = [0] * (3 + pr)
                vec[-1] = 1
                new.append(vec)
        t = 0
        for onset, offset, pitch in s:
            #print(onset, " ", offset, " ", pitch)
            while t < offset:
                one_hot = [0] * (2 + pr) if is_target else [0] * (3 + pr)
                if t < onset:
                    one_hot[0] = 1
                elif t == onset:
                    index = 1 if pitch == -1 else (pitch - pitch_range[0] + 2) #subtract min_pitch
                    #print(index)
                    one_hot[index] = 1
                else:
                    one_hot[0] = 1
                new.append(one_hot)
                t += time_step
        encoded.append(new)
    return encoded


def pad(X, Y, X_len, y_len):
    max_len = len(max(X, key=len))
    for i in range(len(X)):
        empty_X = [0] * X_len
        empty_y = [0] * y_len
        diff = max_len - len(X[i])
        # destructively modifies X, Y
        X[i] += [empty_X] * diff
        Y[i] += [empty_y] * diff
    return max_len


# REQUIRES: pitch_range has form [min, max)
def get_data(directory, time_step, pitch_min, pitch_max, start):
    seqs, pitch_min, pitch_max = read(directory, time_step, pitch_min, pitch_max)
    #print("min ", pitch_min)
    #print("max ", pitch_max)
    X = encode(deepcopy(seqs), time_step, (pitch_min, pitch_max), False, start)
    Y = encode(seqs, time_step, (pitch_min, pitch_max), True)
    for i in range(len(X)):
        X[i] = X[i][:-1] # doesn't give last note
        #Y[i] = Y[i][start:] # removes start tokens
    #max_len = pad(X, Y, (pitch_min, pitch_max))
    #X = np.asarray(X)
    #Y = np.asarray(Y)
    split = train_test_split(X, Y, test_size=0.2)
    return split #, max_len

