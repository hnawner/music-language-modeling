#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split as tts

r_dict = {125:0, 166:1, 167:1, 250:2, 333:3, 334:4, 375:5, 500:6, 666:7, 667:7, 750:8, 833:37, 1000:9, 1250:10,
            1333:11, 1334:11, 1500:12, 1666:13, 1667:14, 1750:15, 2000:16, 2250:17, 2333:18, 2334:18,
            2500:19, 2666:20, 2667:20, 2750:21, 3000:22, 3500: 23, 3750:39, 4000:24, 4500: 38, 5000:25, 6000:26, 7000:27,
            8000:28,9000:29, 10000:30, 11000:31, 12000:32, 13000:33, 14000:34, 15000:35, 16000: 36}

def read_files(folder):
    files = os.listdir(folder)

    mels = []
    rhys = []

    offsets = { "C":0, "C-sharp":1, "D-flat":1, "D":2, "D-sharp":3, "E-flat":3,
                "E":4, "E-sharp":5, "F-flat":4, "F":5, "F-sharp":6, "G-flat":6,
                "G":7, "G-sharp":8, "A-flat":8, "A":9,
                "A-sharp":10, "B-flat":10, "B":11, "B-sharp":0, "C-flat":11 }

    for f in files:
        path = folder + "/" + f
        with open(path, 'r', 0) as f:
            mel = []
            rhy = []
            for line in f:
                parsed = line.split() # delimiter as spaces

                if parsed[0] == "Note":

                    pitch = int(float(parsed[3]))
                    onset = int(float(parsed[1]))
                    offset = int(float(parsed[2]))
                    length = offset - onset
                    if rhy == []: # starts with rest
                        if onset != 0:
                            rhy.append([0, onset])
                            mel.append(1) # rest token
                    elif onset > rhy[-1][1]: # rest
                        rhy.append([rhy[-1][1], onset])
                        mel.append(1) # rest token

                    rhy.append([onset, offset])
                    mel.append(pitch)
               
            rhys.append(rhy)
            mels.append(mel)
    
    return mels, rhys


def setup_rnn(folder, start):
    mels, rhys = read_files(folder)
    X_p = pitch_X_encoder(mels, start)
    y_p = pitch_y_encoder(mels)
    X_r = rhythm_X_encoder(rhys, start)
    y_r = rhythm_y_encoder(rhys)
    for i in range(len(mels)):
        X_p[i] = X_p[i][:-1]
        X_r[i] = X_r[i][:-1]
    X_tr_r, X_te_r, y_tr_r, y_te_r, X_tr_p, X_te_p, y_tr_p, y_te_p = tts(X_r, y_r, X_p, y_p, test_size=0.2)
    return X_tr_r, X_te_r, y_tr_r, y_te_r, X_tr_p, X_te_p, y_tr_p, y_te_p


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


def pitch_X_encoder(X, start):
    X_list = []
    for inst in X:
        start_token = [1] + ([0] * 20)
        vecs = [start_token] * start 
        for pitch in inst:
            start_vec = [0]
            pc_vec = [0] * 12
            octave_vec = [0] * 8
            pc_vec[(pitch % 12)] = 1
            octave_vec[int(pitch / 12)] = 1
            total_vec = start_vec + pc_vec + octave_vec
            vecs.append(total_vec)

        X_list.append(vecs)

    return X_list

def pitch_y_encoder(y):
    y_list = []
    for inst in y:
        mel_vecs = []
        for pitch in inst:
            vector = [0] * 95
            if(pitch != 0):
                vector[int(pitch)] = 1
            mel_vecs.append(vector)


        y_list.append(mel_vecs)

    return y_list


def rhythm_X_encoder(X, start):
    X_list = []
    for inst in X:
        start_token = [1] + ([0] * 40)
        vecs = [start_token] * start 
        for rhy in inst:
            length = int(rhy[1] - rhy[0])
            value = r_dict[length]
            start_vec = [0]
            rhy_vec = [0] * 40
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
            value = r_dict[length]
            rhy_vec = [0] * 40
            rhy_vec[value] = 1
            vecs.append(rhy_vec)

        y_list.append(vecs)

    return y_list
    

#folder = "/home/aeldrid2/REU/krn_split/converted/train/"
#start = 4
#setup_rnn(folder, start)
   
