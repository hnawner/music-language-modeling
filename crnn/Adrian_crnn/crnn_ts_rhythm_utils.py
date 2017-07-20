#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from os import listdir
from copy import deepcopy
from sklearn.model_selection import train_test_split

def read(directory):
    ts = 250 # time step
    seqs = []
    mels = []
    files = []
    for fi in listdir(directory):
        if fi.endswith(".txt"):
            path = directory + '/' + fi
            with open(path) as f:
                seq = []
                mel = []
                replace_length = 0
                replace_onset = None
                replace_pitch = None
                for line in f:
                    line = line.split()

                    if line[0] == "Note":
                        onset = int(float(line[1])) 
                        offset = int(float(line[2]))
                        length = offset - onset
                        pitch = int(float(line[3]))
                        if seq == [] and replace_length == 0: # first note
                            if onset != 0: # starts with rest
                                seq.append([0, onset])
                                mel.append(2)
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
                                        seq.append([replace_onset, (replace_onset + replace_length)])
                                        mel.append(replace_pitch)
                                        replace_pitch = None
                                        replace_onset = None
                                        replace_length = 0
                                    else:
                                        print("rest HELP ME!!")  
                                        print(fi) 
                                seq.append((seq[-1][1], onset))
                                mel.append(2) # rest token
                        
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
                                    seq.append([replace_onset, (replace_onset + replace_length)])
                                    mel.append(replace_pitch)
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
                            seq.append((onset, offset))
                            mel.append(pitch)
                        if replace_length !=0 and replace_length % ts == 0:
                            seq.append([replace_onset, (replace_onset + replace_length)])
                            mel.append(replace_pitch)
                            replace_pitch = None
                            replace_onset = None
                            replace_length = 0
                seqs.append(seq)
                mels.append(mel)
                files.append(fi)
    return seqs, mels, files




def encode_input(seqs, files, time_step, kernel_size):
    # INDICES: LEN 2
    # 0: note is playing
    # 1: note is articulated
    # 2: start of sequence
    encoded = []  
    count = 0
    for s, fi in zip(seqs,files):
        new = [[0, 0, 1] for i in range(kernel_size)] # start of sequence
        end = int(s[-1][1])
        for t in range(0, end, int(time_step)):
            if t < s[0][0]:
                new.append([0,0,0])
            elif t == s[0][0]:
                new.append([1,1,0])
            elif t > s[0][0] and t < s[0][1]:
                new.append([1,0,0])
            elif t >= s[0][1] and t < s[1][0]:
                new.append([0,0,0])
                s.pop(0)
            elif t == s[1][0]:
                new.append([1,1,0])
                s.pop(0)
            else:
                print('inputs')
                print('time step', t)
                print(fi)
                print('s[0]', s[0])
                print('s[1]', s[1])
                print('count', count)
                raise Exception("nonexhaustive match failure")
        count += 1
        encoded.append(new)
    return encoded


def encode_target(seqs, files, time_step):
    # INDICES: LEN 3
    # 0: note is articulated
    # 1: note is playing
    encoded = []
    count = 0
    for s, fi in zip(seqs, files):
        new = []
        end = int(s[-1][1])
        for t in range(0, end, int(time_step)):
            if t < s[0][0]:
                new.append([0,0])
                print("1uh oh")
                print(s[0][0])
                print(fi)
            elif t == s[0][0]:
                new.append([1,0])
            elif t > s[0][0] and t < s[0][1]:
                new.append([0,1])
            elif t >= s[0][1] and t < s[1][0]:
                new.append([0,0])
                print("2uh oh")
                print(fi)
                s.pop(0)
            elif t == s[1][0]:
                new.append([1,0])
                s.pop(0)
            else:
                print('targets')
                print('time step', t)
                print(fi)
                print('s[0]', s[0])
                print('s[1]', s[1])
                raise Exception("nonexhaustive match failure")
        count += 1
        encoded.append(new)
    return encoded


def pad(X, Y):
    max_len = len(max(X, key=len))
    for i in range(len(X)):
        diff = max_len - len(X[i])
        # destructively modifies X, Y
        X[i] += [[-1,-1,-1]] * diff
        Y[i] += [[0,0]] * diff
    return max_len

def get_lengths(L):
    lens = deepcopy(L)
    return map(lambda l: len(l), L)

def get_data(directory, time_step, kern_size):
    seqs, mels = read(directory)
    X = encode_input(deepcopy(seqs), time_step)
    Y = encode_target(seqs, time_step)
    for i in range(len(X)):
        X[i] = X[i][:-1]
        Y[i] = Y[i][int(kern_size):]
    max_len = pad(X, Y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    split = train_test_split(X, Y, test_size=0.2)
    return split, max_len

'''
seqs, mels, files = read("/home/aeldrid2/REU/krn_split/converted/train/")

X = seqs[:-1]
X = encode_input(deepcopy(X), files, 125, 8)
Y = encode_target(seqs, files, 125)
print(X[7])
print(Y[7])
'''
