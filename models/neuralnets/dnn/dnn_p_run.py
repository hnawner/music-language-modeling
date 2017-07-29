#!/usr/bin/env python

from dnn import DNN
import sys

i = int(sys.argv[1])

train = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_train/"
test = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_test/"

t = True
m = "default"
if i == 2:
    t = False
elif i == 3:
    m = "bn"
elif i == 4:
    m = "ls"


model = DNN(train, test, 8, "pitch", model_type = m, trans = t)
