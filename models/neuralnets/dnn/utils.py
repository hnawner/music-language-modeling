#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
import numpy as np
from os import listdir


def read_mels(folder, trans = True):
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

                if trans and parsed[0] == "*K":
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


def make_ngrams(seqs, n):
    grams = []
    for seq in seqs:
        prevs = seq[:(n-1)]
        for index in range((n-1), len(seq)):
            prevs += [ (seq[index]) ]
            grams.append(prevs)
            prevs = prevs[1:]
    return grams


def pitch_encode(grams, p_min, pr, encode):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []
        for g in gram[:-1]:
        	if encode == "abs":
        		pitch = [0] * pr
        		pitch[(g - p_min)] = 1
        		vecs += pitch
        	elif encode == "pc":
        		pc = [0] * 12
				octave = [0] * 8
				pc[g % 12] = 1
				octave[g // 12] = 1
				vecs += pc + octave
        target = gram[-1] - p_min
        vecs_list.append(vecs)
        targets.append(target)

    return vecs_list, targets

def rhythm_encode(grams):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []	
        for rhy in gram[:-1]:
            length = int(rhy[1] - rhy[0])
            value = add_to_r_dict(length, False, False)
            rhy_vec = [0] * len(r_dict)
            rhy_vec[value] = 1
            vecs += rhy_vec
        target = add_to_r_dict((gram[-1][1] - gram[-1][0]), False, False)
        targets.append(target)
        vecs_list.append(vecs)
    

    return vecs_list, targets


def setup(tr_folder, te_folder, n, d_type, p_min = 40, pr = 55, trans = True, encode = "pc"):
    tr_mels, te_mels = read_mels(tr_folder, trans), read_mels(te_folder, trans)
    if d_type == "rhythm":
    	tr_mels, te_mels = read_rhythms(tr_folder, True), \
				read_rhythms(te_folder, False)
    tr_grams, te_grams = make_ngrams(tr_mels, n), make_ngrams(te_mels, n)
    tr_x, tr_y, te_x, te_y = None, None, None, None
    if d_type == "pitch":
    	tr_x, tr_y = pitch_encode(tr_grams, p_min, pr, encode)
    	te_x, te_y = pitch_encode(te_grams, p_min, pr, encode)
    if d_type == "rhythm":
    	tr_x, tr_y = rhythm_encode(tr_grams)
    	te_x, te_y = rhythm_encode(te_grams)
    return tr_x, tr_y, te_x, te_y

