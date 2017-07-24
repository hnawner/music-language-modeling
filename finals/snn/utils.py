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
                    pass
                    # offset = offsets[parsed[2]]

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


def encode(grams, p_min, pr):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []
        for g in gram[:-1]:
            pc = [0] * 12
            octave = [0] * 8
            pc[g % 12] = 1
            octave[g // 12] = 1
            vecs += pc + octave
            #vec = [0] * pr
            #vec[g - p_min] = 1
            #vecs += vec
        target = gram[-1] - p_min
        vecs_list.append(vecs)
        targets.append(target)

    return vecs_list, targets


def setup(tr_folder, te_folder, n, p_min, pr):
    tr_mels, te_mels = read_files(tr_folder), read_files(te_folder)
    tr_grams, te_grams = make_ngrams(tr_mels, n), make_ngrams(te_mels, n)
    tr_x, tr_y = encode(tr_grams, p_min, pr)
    te_x, te_y = encode(te_grams, p_min, pr)
    #print('tr_x', tr_x[0])
    #print('tr_y', tr_y[0])
    return tr_x, tr_y, te_x, te_y


#setup('/home/hawner2/reu/musical-forms/mels/pitches/folk_major/', '/home/hawner2/reu/musical-forms/mels/pitches/folk_maj_test/', 8, 45, 50)
