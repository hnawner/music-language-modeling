#!/usr/bin/env python

from rnn import RNN
from dnn import DNN
from crnn import CRNN
import sys

i = int(sys.argv[1])

model = None
if i == 1 or i == 2:
    model = RNN
if i == 3 or i == 4:
    model = CRNN
if i == 5 or i == 6:
    model = DNN

train = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_train/"
test = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_test/"

model_type = "combine"
if i % 2 == 0:
    model_type = "separate"

transpose = True

run = model(train, test, model_type, transpose)
