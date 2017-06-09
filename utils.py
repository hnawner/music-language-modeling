#!/usr/bin/env python

import os
import numpy as np

def read_files(folder):
    files = os.listdir(folder)

    maj_mels = []
    min_mels = []

    offsets = { "C":0, "C-sharp":1, "D-flat":1, "D":2, "D-sharp":3, "E-flat":3,
                "E":4, "E-sharp":5, "F-flat":4, "F":5, "F-sharp":6, "G-flat":6,
                "G":7, "G-sharp":8, "A-flat":8, "A":9,
                "A-sharp":10, "B-flat":10, "B":11, "B-sharp":0, "C-flat":11 }

    for f in files:
        path = folder + "/" + f
        offset = 0 # offset from key of C
        is_major = True # default
        with open(path, 'r', 0) as f:
            mel = []
            for line in f:
                parsed = line.split() # delimiter as spaces

                if parsed[0] == "Info" and parsed[1] == "key":
                    if parsed[3] == "Minor": is_major = False
                    offset = offsets[parsed[2]]

                elif parsed[0] == "Note":
                    pitch = int(parsed[3]) - offset
                    mel.append(pitch)

            if is_major: maj_mels.append(mel)
            else: min_mels.append(mel)

    maj_mels_np = np.asarray(maj_mels)
    min_mels_np = np.asarray(min_mels)
    
    return maj_mels_np, min_mels_np

