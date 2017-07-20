#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split as tts
import pick_up_parse

offsets = { "C":0, "C-sharp":1, "D-flat":1, "D":2, "D-sharp":3, "E-flat":3,
            "E":4, "E-sharp":5, "F-flat":4, "F":5, "F-sharp":6, "G-flat":6,
            "G":7, "G-sharp":8, "A-flat":8, "A":9,
            "A-sharp":10, "B-flat":10, "B":11, "B-sharp":0, "C-flat":11 }
            
#meters = {"simple duple" : 0, "simple triple" : 0, "simple quadruple" : 0, 
#            "compound duple" : 1, "compound triple" : 1, "compound quadruple": 1,
#            "other" : 2}

meter_div = {"simple": 0, "compound": 1, "other": 2}
meter_beat = {"duple": 0, "triple": 1, "quadruple": 0, "other": 2}

                
'''
def read_files(folder, data_type):
    files = os.listdir(folder)
    
    pick_up_dict = pick_up_parse.build_pick_up_dict("/Volumes/A3 Personal/REU/musical-forms/mels/krnfiles")

    mels = []
    mel_classes = []

    for fi in files:
        path = folder + "/" + fi
        with open(path, 'r', 0) as f:
            mel = []
            for line in f:
                parsed = line.split() # delimiter as spaces

                if data_type == "key" and parsed[0] == "Info" and parsed[1] == "key":
                    offset = offsets[parsed[2]]
                    mode = 1
                    if parsed[3] == "Minor":
                        mode = 0
                    key = [offset, mode]
                    mel_classes.append(key)
                    
                elif data_type == "meter" and parsed[1] == "AMT:":
                    if len(parsed) >= 4 and (parsed[2]+" "+parsed[3]) in meters:
                        met = int(meters[parsed[2]+" "+parsed[3]])
                    else:
                        met = int(meters["other"])
                    off = pick_up_dict[fi]
                    mel_classes.append([met, off])


                elif parsed[0] == "Note":
                    if data_type == "key":
                        pitch = int(parsed[3])
                        mel.append(pitch)
                    elif data_type == "meter":
                        start_time = int(parsed[1])
                        mel.append(start_time)
                    
            mels.append(mel)

    
    return mels, mel_classes
'''

pickups = {}

def read_files(folder, data_type):
    files = os.listdir(folder)
    
    
    mels = []
    mel_classes = []

    for fi in files:
        path = folder + "/" + fi
        with open(path, 'r', 0) as f:
            mel = []
            beat = None
            div = None
            pickup = None
            for line in f:
                parsed = line.split() # delimiter as spaces

                if data_type == "key" and parsed[0] == "*K":
                    offset = offsets[parsed[1]]
                    mode = 1
                    if parsed[2] == "Minor":
                        mode = 0
                    key = [offset, mode]
                    mel_classes.append(key)
                    
                elif data_type == "meter" and parsed[0] == "*M":
                    if len(parsed) >= 3:

                        if parsed[1] in meter_div:
                            div = meter_div[parsed[1]]
                        else:
                            div = meter_div["other"]
                        if parsed[2] in meter_beat:
                            beat = meter_beat[parsed[2]]
                        else:
                            beat = meter_beat["other"]                          
                    else:
                        beat = int(meter_beat["other"])
                        div = int(meter_div["other"])
                        

                    
                elif data_type == "meter" and parsed[0] == "*P":
                    pickup = int(float(parsed[1]))


                elif parsed[0] == "Note":
                    if data_type == "key":
                        pitch = int(parsed[3])
                        mel.append(pitch)
                    elif data_type == "meter":
                        start_time = int(float(parsed[1]))
                        mel.append(start_time)
                    
            mels.append(mel)
            if data_type == "meter":
            	measure_length = 0
            	if beat == 0:
            		measure_length = 2
            	elif beat == 1:
            		measure_length = 3
            	if div == 0:
            		measure_length *= 1000
            	elif div == 1:
            		measure_length *= 1500
            	if measure_length != 0:
            		pickup = (pickup % measure_length)
            	
                mel_classes.append([div, beat, pickup])

                if div != 2 and beat != 2:
                    if (pickup in pickups) == False:
                        pickups[pickup] = 1
                        print(fi, " ", pickup)
                    else:
                        pickups[pickup] += 1
    print(pickups)
    return mels, mel_classes
    

    
def transpose(mels, keys):
    transposed = []
    for i in range(len(mels)):
        offset = (keys[i])[0]
        transposed_mel = [(p - offset) % 12 for p in mels[i]]
        transposed.append(transposed_mel)
        
    return transposed
    
    

def make_id_data(mels, mel_classes, length):
    X = []
    y = []
    for mel, c in zip(mels, mel_classes):
        if len(mel) >= length:
            X.append(mel[:length])
            y.append(c)
            
    return X, y



