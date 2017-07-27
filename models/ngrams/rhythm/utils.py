#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split as tts

r_dict = {"unkown": 0}

def read_files(folder):
    files = os.listdir(folder)

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
                            new_rhy = add_to_r_dict((onset))
                            rhy.append([0, onset])
                    elif onset > rhy[-1][1]: # rest
                        new_rhy = add_to_r_dict((onset - rhy[-1][1]))
                        rhy.append([rhy[-1][1], onset])
                        
                    new_rhy = add_to_r_dict(length)
                    rhy.append([onset, offset])
                                
            rhys.append([add_to_r_dict(off - on) for on, off in rhy])
    
    return rhys


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
