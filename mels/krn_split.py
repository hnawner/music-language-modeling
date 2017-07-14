#!/usr/bin/env python

import os
import numpy as np
import sys

def split():
	test_folder = os.listdir("key-meter-id/test/")
	test_list = [f for f in test_folder]
	
	krn_folder = os.listdir("krnfiles")
	
	for f in krn_folder:
		path = "krnfiles/" + f
		print(path)
		with open(path, "r", 0) as fi:
			if f in test_list:
				n_path = "krn_split/krn/test/" + f
			else:
				n_path = "krn_split/krn/train/" + f
			n = open(n_path, 'w+')
			for line in fi:
				n.write(line)
			n.close()
			
	print("Done!")
	
split()