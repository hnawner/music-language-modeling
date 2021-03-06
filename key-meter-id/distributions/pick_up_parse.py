#!/usr/bin/env python

import os
import numpy as np
import sys

def build_pick_up_dict(folder):
    files = os.listdir(folder)

    pick_up_dict = {}

    for fi in files:
        path = folder + "/" + fi
        with open(path, 'r', 0) as f:
            pick_ups = []
            p_indicator = False
            for line in f:
                if p_indicator == False:
                    if ("!" in line) == False and ("*" in line) == False:
                        p_indicator = True
                if p_indicator:
                    if ("=1" in line) == False:
                        pick_ups.append(line)
                    elif "=1" in line:
                        p_indicator = False
                        break
            #print(f, pick_ups)
            pickup_val = parse_note_vals(pick_ups)
            if pickup_val == -1:
            	print(fi)
            pick_up_dict[fi] = pickup_val
     
    #print(pick_up_dict)
    return pick_up_dict
            
            
def parse_note_vals(notes):
    note_vals = {16: 250, 8: 500, 4: 1000, 2: 2000, 1: 4000, 0: 1600, 12: 333}
    total_val = 0
    for note in notes:
    	found = False
        for k in note_vals.keys():
            if str(k) in note:
                total_val += note_vals[k]
                found= True
                break
        if found == False:
        	total_val = -1
        	print("Problem ", note)
                
    return total_val
    
    
    
            