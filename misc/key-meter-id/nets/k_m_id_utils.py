#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split as tts
import math

n_rhythms = 28
n_pitches = 88


def read_files(folder, data, r_dict = None):
    files = os.listdir(folder)

    mels = []
    mel_class = []

    offsets = { "C":0, "C-sharp":1, "D-flat":1, "D":2, "D-sharp":3, "E-flat":3,
                "E":4, "E-sharp":5, "F-flat":4, "F":5, "F-sharp":6, "G-flat":6,
                "G":7, "G-sharp":8, "A-flat":8, "A":9,
                "A-sharp":10, "B-flat":10, "B":11, "B-sharp":0, "C-flat":11 }
                
    meters = {"simple duple" : 0, "simple triple" : 1, "simple quadruple" : 2, 
    		  "compound duple" : 3, "compound triple" : 4, "compound quadruple": 5}
                

    for f in files:
        path = folder + "/" + f
        offset = 0 # offset from key of C
        with open(path, 'r', 0) as f:
            mel = []
            for line in f:
                parsed = line.split() # delimiter as spaces

                if data = "pitch" and parsed[0] == "Info" and parsed[1] == "key":
                    offset = offsets[parsed[2]]
                    mel_class.append(offset)
                elif data = "rhythm" and parsed[1] = "AMT":
                	keyword = str(parsed[2]) + " " + str(parsed[3])
                	meter = meters[keyword]
                	mel_class.append(meter)
                	

                elif parsed[0] == "Note":
                	# pitch
                    pitch = int(parsed[3]) - int(offset)
                    
                    # rhythm
                    length = None
                    if f[i+1].split()[0] == "Note":
                        length = int(f[i+1].split()[1]) - int(parsed[1])
                    else:
                        length = int(parsed[2]) - int(parsed[1])
                    
                    r_approx = [v for (k, v) in r_dict.items() 
                        if abs(length - k) < 5]
                    rhythm = r_approx[0]
                    
                    if data == "pitch":
                    	mel.append(pitch)
                    elif data == "rhythm":
                        mel.append(rhythm)


            mels.append(mel)
                            
    return mels, mel_class
    
    
    
def build_rhythm_dict(folder):
    data = os.listdir(folder)
    
    rhythms = []
    
    for f in data:
        #if ".txt" in f:
            f_path = folder + "/" + f
            with open(f_path, 'r', 0) as f:
            	f = [line for line in f]
                for i in range(len(f)):
                    parsed = f[i].split("\t")

                    if parsed[0] == "Note":
                        length = None
                        if f[i+1].split()[0] == "Note":
                            length = int(f[i+1].split()[1]) - int(parsed[1])
                        else:
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
    
    
    
def make_id_data(mels, length):
	data = []
	for mel in mels:
		data.append[mel[:length]]
		
    return data
    

def rnn_encoder(X):
    X_list = []
    for inst in X:
        vecs = []
        for pitch in inst:
            pc_vec = [0] * 12
            octave_vec = [0] * 8
            pc_vec[(pitch % 12)] = 1
            octave_vec[(pitch / 12)] = 1
            total_vec = pc_vec + octave_vec
            vecs.append(total_vec)

        X_list.append(vecs)

    return X_list
	
	
def setup(folder, length, id_type):
	r_dict = None
	if id_type = "rhythm":
		r_dict = build_rhythm_dict(folder)
    X, y = read_files(folder, id_type, r_dict)
    X = make_id_data(X, length)
    X = rnn_encoder(X)
    X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test

	
	
	