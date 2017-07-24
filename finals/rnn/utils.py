#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
import numpy as np
from os import listdir


def read_files(folder):
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

                if parsed[0] == "Info" and parsed[1] == "key":
                    offset = offsets[parsed[2]]

                elif parsed[0] == "Note":
                    pitch = int(parsed[3]) - offset
                    mel.append(pitch)

            mels.append(mel)
    
    return mels

def pad(tr_mels, te_mels):
    max_len = len(max(tr_mels + te_mels, key=len))
    for mels in [tr_mels, te_mels]:
        for m in mels:
            diff = max_len - len(m)
            padding = [-1] * diff
            m += padding
    return max_len - 1

def encode(mels, p_min, pr):
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

def setup(tr_folder, te_folder, p_min, pr):
    tr_mels, te_mels = read_files(tr_folder), read_files(te_folder)
    max_len = pad(tr_mels, te_mels)
    tr_x, tr_y = encode(tr_mels, p_min, pr)
    te_x, te_y = encode(te_mels, p_min, pr)
    return tr_x, tr_y, te_x, te_y, max_len


