#!/usr/bin/env python

from rnn import RNN
from dnn import DNN
from crnn import CRNN

model = CRNN
train = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_train/"
test = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_test/"
model_type = "combine"
transpose = True

run = model(train, test, model_type, transpose)
