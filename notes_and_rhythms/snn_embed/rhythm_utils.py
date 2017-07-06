#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split as tts
import math

n_rhythms = 28
n_pitches = 88


def read_files(folder, r_dict = None, data = "both"):
    files = os.listdir(folder)

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
                    length = int(parsed[2]) - int(parsed[1])
                    
                    if data == "pitch":
                    	mel.append(pitch)
                    elif data == "rhythm":
                    	r_approx = [v for (k, v) in r_dict.items() 
                    	    if abs(length - k) < 5]
                    	rhythm = r_approx[0]
                    	mel.append(rhythm)
                    elif data == "both":
                    	r_approx = [v for (k, v) in r_dict.items() 
                    	    if abs(length - k) < 5]
                    	rhythm = r_approx[0]
                    	note = [pitch, rhythm]
                        mel.append(note)

            mels.append(mel)
                            
    return mels

def read_files_to_embed(folder, r_dict = None, data = "both"):
    files = os.listdir(folder)

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
                    length = int(parsed[2]) - int(parsed[1])             
                    r_approx = [v for (k, v) in r_dict.items() 
                        if abs(length - k) < 5]
                    rhythm = r_approx[0]
                    note = pitch * n_rhythms + rhythm
                    mel.append(note)

            mels.append(mel)
                            
    return mels



    
def build_rhythm_dict(folder):
    data = os.listdir(folder)
    
    rhythms = []
    
    for directory in data:
        path = folder + "/" + directory
        if "folk" in path:
            directory = os.listdir(path)
            for f in directory:
                if ".txt" in f:
                    f_path = path + "/" + f
                    with open(f_path, 'r', 0) as f:
                        for line in f:
                            parsed = line.split()
            
                            if parsed[0] == "Note":
                                length = int(parsed[2]) - int(parsed[1])
                                diffs = [abs(r - length) for r in rhythms]
                                if diffs != [] and min(diffs) < 5:
                                    continue
                                else: rhythms.append(length)
                                
    rhythms.sort()
    r_dict = {}
    for i in range(len(rhythms)):
        r_dict[(rhythms[i])] = i
                    
    return r_dict
    
    
def make_ngrams(seqs, n):
    grams = []
    for seq in seqs:
        prevs = seq[:(n-1)]
        for index in range((n-1), len(seq)):
            prevs += [ (seq[index]) ]
            grams.append(prevs)
            prevs = prevs[1:]
    return grams

def split_inst_targets(grams):
    inst = []
    labels = []
    for gram in grams:
        inst.append(gram[:(len(gram)-1)])
        labels.append(gram[-1])

    return inst, labels

   
    
 
def one_hot_ngram_rhythm(grams):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []
        for index in range(len(gram) - 1):
            vec = [0] * n_rhythms
            vec[(gram[index])] = 1
            vecs += vec
        target = gram[-1]
        vecs_list.append(vecs)
        targets.append(target)

    return vecs_list, targets
    
def one_hot_ngram_pitch_and_rhythm(grams):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []
        for index in range(len(gram) - 1):
            vec = [0] * (n_rhythms * n_pitches)
            vec[((gram[index][0]) * n_rhythms) + (gram[index][1])] = 1
            vecs += vec
        target = (gram[-1][0] * n_rhythms) + gram[-1][1]
        vecs_list.append(vecs)
        targets.append(target)

    return vecs_list, targets


def one_hot_ngram_pitch(grams):
    vecs_list = []
    targets = []
    for gram in grams:
        vecs = []
        for index in range(len(gram) - 1):
            pc_vec = [0] * 12
            octave_vec = [0] * 8
            pc_vec[(gram[index] % 12)] = 1
            octave_vec[(gram[index] / 12)] = 1
            vecs += pc_vec
            vecs += octave_vec
        target = gram[-1]
        vecs_list.append(vecs)
        targets.append(target)

    return vecs_list, targets
    
def setup_ngrams(master_folder, folder, n, data_type):
    encoder = None
    if data_type == "pitch":
        encoder = one_hot_ngram_pitch
    elif data_type == "rhythm":
        encoder = one_hot_ngram_rhythm
    else:
        encoder = one_hot_ngram_pitch_and_rhythm

    r_dict = build_rhythm_dict(master_folder)
    mels = read_files(folder, r_dict, data_type)
    grams = make_ngrams(mels, n)
    X, y = encoder(grams)
    X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test
    
   
   
def debug():
	X_train, X_test, y_train, y_test = setup_ngrams("mels", "mels/folk_major",
		2, "both", one_hot_ngram_pitch_and_rhythm)
	
	print(X_train[3])
	print(y_train[3])
	
	
