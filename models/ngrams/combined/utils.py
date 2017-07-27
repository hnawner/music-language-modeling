#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split as tts

offsets = { "C":0, "C-sharp":1, "D-flat":1, "D":2, "D-sharp":3, "E-flat":3,
            "E":4, "E-sharp":5, "F-flat":4, "F":5, "F-sharp":6, "G-flat":6,
            "G":7, "G-sharp":8, "A-flat":8, "A":9,
            "A-sharp":10, "B-flat":10, "B":11, "B-sharp":0, "C-flat":11 }

r_dict = {"unkown": 0}

def read_files(folder, trans):
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
 
                    onset = int(float(parsed[1]))
                    offset = int(float(parsed[2]))
                    length = offset - onset
                    if rhy == []: # starts with rest
                        if onset != 0:
                            new_rhy = add_to_r_dict((onset))
                            rhy.append([0, onset])
                            mel.append(-1) # rest token                           
                    elif onset > rhy[-1][1]: # rest
                        new_rhy = add_to_r_dict((onset - rhy[-1][1]))
                        rhy.append([rhy[-1][1], onset])
                        mel.append(-1) # rest token
                        
                    new_rhy = add_to_r_dict(length)
                    rhy.append([onset, offset])
                    mel.append(pitch)
                                
            rhys.append([add_to_r_dict(off - on) for on, off in rhy])
            mels.append(mel)
    
    return mels, rhys


def add_to_r_dict(length):
    if length in r_dict:
        return r_dict[length]
    else:
        # for triplet slopiness
        if (length + 1) in r_dict:
            return r_dict[length + 1]
        elif (length - 1) in r_dict:
            return r_dict[length -1]
        else:
            r_dict[length] = len(r_dict)
            return r_dict[length]




