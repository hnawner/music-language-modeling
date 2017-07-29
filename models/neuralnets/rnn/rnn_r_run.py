#!/usr/bin/env python

from rnn import RNN
import sys

i = int(sys.argv[1])

n = 128
if i == 2:
    n = 256
if i == 3:
    n = 512

train = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_train/"
test = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_test/"

model = RNN(train, test, n, "rhythm")
