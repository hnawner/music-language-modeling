#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
import numpy as np
from os import listdir


def read_mels(folder):
    files = listdir(folder)

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

                if parsed[0] == "*K":
                    offset = offsets[parsed[1]]

                elif parsed[0] == "Note":
                    pitch = int(float(parsed[3])) - offset
                    mel.append(pitch)

            mels.append(mel)
    
    return mels
    
r_dict = {"unknown": 0}

def read_rhythms(folder, train):
    files = listdir(folder)

    rhys = []

    for f in files:
        path = folder + "/" + f
        with open(path, 'r', 0) as f:
            rhy = []
            key_offset = 0
            for line in f:
                parsed = line.split() # delimiter as spaces

                if parsed[0] == "Note":

                    onset = int(float(parsed[1]))
                    offset = int(float(parsed[2]))
                    length = offset - onset
                    if rhy == []: # starts with rest
                        if onset != 0:
                            rhy.append([0, onset])
                            add_to_r_dict((onset), True, train)
                    elif onset > rhy[-1][1]: # rest
                        rhy.append([rhy[-1][1], onset])
                        add_to_r_dict((onset - rhy[-1][1]), True, train)

                    rhy.append([onset, offset])
                    add_to_r_dict(length, True, train)

            rhys.append(rhy)
    
    return rhys


def add_to_r_dict(length, read, train):
    if length in r_dict:
        return r_dict[length]
    else:
        # for triplet slopiness
        if (length + 1) in r_dict:
            return r_dict[length + 1]
        elif (length - 1) in r_dict:
            return r_dict[length -1]
        elif read and train: # if reading train data
            r_dict[length] = len(r_dict)
            return r_dict[length]
        else: # don't alter dict of encoding or reading test data
            return r_dict["unknown"]
            
            

def pad(tr_mels, te_mels):
    max_len = len(max(tr_mels + te_mels, key=len))
    for mels in [tr_mels, te_mels]:
        for m in mels:
            diff = max_len - len(m)
            padding = [-1] * diff
            m += padding
    return max_len - 1

def pitch_encode(mels, p_min, pr):
    inputs = []
    labels = []
    for m in mels:
        vecs = []
        targets = []
        for n in m:
            if n >= 0:
                pc_vec = [0] * 12
                octave_vec = [0] * 8
                pc_vec[n % 12] = 1
                octave_vec[n // 12] = 1
                vecs.append(pc_vec + octave_vec)
                target = [0] * pr
                target[n - p_min] = 1 
                targets.append(target)
            else:
                vecs.append([0] * 20)
                targets.append([0] * pr)
        inputs.append(vecs[:-1])
        labels.append(targets[1:])

    return inputs, labels
    
def rhythm_encode(mels):
    inputs = []
    labels = []
    for m in mels:
        vecs = []
        targets = []
        for rhy in m:
            if type(rhy) == list:
                length = int(rhy[1] - rhy[0])
                value = add_to_r_dict(length, False, False)
                vec = [0] * len(r_dict)
                vec[value] = 1
                vecs.append(vec)
            else: # padding
                vec = [0] * len(r_dict)
                vecs.append(vec)
        inputs.append(vecs[:-1])
        labels.append(vecs[1:])

    return inputs, labels

def setup(tr_folder, te_folder, d_type, p_min, pr):
    tr_mels, te_mels = None, None
    if d_type == "pitch":
        tr_mels, te_mels = read_mels(tr_folder), read_mels(te_folder)
    elif d_type == "rhythm":
        tr_mels, te_mels = read_rhythms(tr_folder, True), read_rhythms(te_folder, False)
    max_len = pad(tr_mels, te_mels)
    tr_x, tr_y, te_x, te_y = None, None, None, None
    if d_type == "pitch":
    	tr_x, tr_y = pitch_encode(tr_mels, p_min, pr)
    	te_x, te_y = pitch_encode(te_mels, p_min, pr)
    elif d_type == "rhythm":
    	tr_x, tr_y = rhythm_encode(tr_mels)
    	te_x, te_y = rhythm_encode(te_mels)
    return tr_x, tr_y, te_x, te_y, max_len


