#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split as tts

offsets = { "C":0, "C-sharp":1, "D-flat":1, "D":2, "D-sharp":3, "E-flat":3,
            "E":4, "E-sharp":5, "F-flat":4, "F":5, "F-sharp":6, "G-flat":6,
            "G":7, "G-sharp":8, "A-flat":8, "A":9,
            "A-sharp":10, "B-flat":10, "B":11, "B-sharp":0, "C-flat":11 }

r_dict = {"unknown": 0}

def read_files(folder, p_max, p_min, trans, train):
    files = os.listdir(folder)

    mels = []
    rhys = []

    for f in files:
        path = folder + "/" + f
        with open(path, 'r', 0) as f:
            mel = []
            rhy = []
            key_offset = 0
            for line in f:
                parsed = line.split() # delimiter as spaces

                if trans and parsed[0] == "*K":
                    key_offset = offsets[str(parsed[1])]

                elif parsed[0] == "Note":

                    pitch = int(float(parsed[3])) - key_offset
                    if train and pitch < (p_min + 2):
                        p_min = pitch - 2
                    elif train and pitch > (p_max - 1):
                        p_max = pitch + 1
                    onset = int(float(parsed[1]))
                    offset = int(float(parsed[2]))
                    length = offset - onset                    
                    if rhy == []: # starts with rest
                        if onset != 0:
                            rhy.append([0, onset])
                            mel.append(-1) # rest token
                            add_to_r_dict((onset), True, train)                            
                    elif onset > rhy[-1][1]: # rest
                        rhy.append([rhy[-1][1], onset])
                        mel.append(-1) # rest token
                        add_to_r_dict((onset - rhy[-1][1]), True, train)

                    rhy.append([onset, offset])
                    mel.append(pitch)
                    add_to_r_dict(length, True, train)
               
            rhys.append(rhy)
            mels.append(mel)
    
    return mels, rhys, p_max, p_min


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
            


def setup_test(folder, p_max, p_min, start, trans):
    mels, rhys, _, _ = read_files(folder, p_max, p_min, trans, False)
    X_p = pitch_X_encoder(mels, start, p_max, p_min)
    y_p = pitch_y_encoder(mels, p_max, p_min)
    X_r = rhythm_X_encoder(rhys, start)
    y_r = rhythm_y_encoder(rhys)
    for i in range(len(mels)):
        X_p[i] = X_p[i][:-1]
        X_r[i] = X_r[i][:-1]
    return X_p, y_p, X_r, y_r


def setup_rnn(folder, start, trans):
    p_max, p_min = 80, 40 # intital range
    mels, rhys, p_max, p_min = read_files(folder, p_max, p_min, trans, True)
    X_p = pitch_X_encoder(mels, start, p_max, p_min)
    y_p = pitch_y_encoder(mels, p_max, p_min)
    X_r = rhythm_X_encoder(rhys, start)
    y_r = rhythm_y_encoder(rhys)
    for i in range(len(mels)):
        X_p[i] = X_p[i][:-1]
        X_r[i] = X_r[i][:-1]
    X_tr_r, X_te_r, y_tr_r, y_te_r, X_tr_p, X_te_p, y_tr_p, y_te_p = \
            tts(X_r, y_r, X_p, y_p, test_size=0.2)
    return (X_tr_r, X_te_r, y_tr_r, y_te_r, X_tr_p, X_te_p, y_tr_p, y_te_p), \
                                                             (p_max, p_min)


def max_length(mels):
    max_len = 0
    for mel in mels:
        if len(mel) > max_len:
            max_len = len(mel)

    return max_len



def pad(X, Y, x_len, y_len):
    max_len = len(max(X, key=len))
    for i in range(len(X)):
        diff = max_len - len(X[i])
        # destructively modifies X, Y
        X[i] += [[0] * x_len] * diff
        Y[i] += [[0] * y_len] * diff
    return max_len

def get_mel_lengths(mels):
    lengths = [len(mel) for mel in mels]
    return lengths


def pitch_X_encoder(X, start, p_max, p_min):
    pr = p_max - p_min
    X_list = []
    for inst in X:
        start_token = [1] + ([0] * 18)
        vecs = [start_token] * start 
        for pitch in inst:
            pitch = int(pitch)
            start_vec = [0]
            rest_vec = [0]
            pc_vec = [0] * 12
            octave_vec = [0] * 5
            if pitch == -1: #rest
                rest_vec[0] = 1
            else: # pitch
                pitch = pitch - p_min
                if pitch >= 0 and pitch < p_max:
                    pc_vec[(pitch % 12)] = 1
                    octave_vec[(pitch / 12)] = 1
            total_vec = start_vec + rest_vec + pc_vec + octave_vec
            vecs.append(total_vec)

        X_list.append(vecs)

    return X_list

def pitch_y_encoder(y, p_max, p_min):
    pr = p_max - p_min
    y_list = []
    for inst in y:
        mel_vecs = []
        for pitch in inst:
            pitch = int(pitch)
            vector = [0] * pr
            if pitch == -1:
                vector[0] = 1 # rest token
            pitch = pitch - p_min
            if pitch >= 0 and pitch < p_max:
                vector[pitch] = 1
            mel_vecs.append(vector)

        y_list.append(mel_vecs)

    return y_list


def rhythm_X_encoder(X, start):
    X_list = []
    for inst in X:
        start_token = [1] + ([0] * len(r_dict))
        vecs = [start_token] * start 
        for rhy in inst:
            length = int(rhy[1] - rhy[0])
            value = add_to_r_dict(length, False, False)
            start_vec = [0]
            rhy_vec = [0] * len(r_dict)
            rhy_vec[value] = 1
            total_vec = start_vec + rhy_vec
            vecs.append(total_vec)
        X_list.append(vecs)

    return X_list

def rhythm_y_encoder(y):
    y_list = []
    for inst in y:
        vecs = []
        for rhy in inst:
            length = int(rhy[1] - rhy[0])
            value = add_to_r_dict(length, False, False)
            rhy_vec = [0] * len(r_dict)
            rhy_vec[value] = 1
            total_vec = rhy_vec
            vecs.append(total_vec)

        y_list.append(vecs)

    return y_list
    
   
