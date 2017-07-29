#!/usr/bin/env python

from dnn import DNN
import sys

i = int(sys.argv[1])

train = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_train/"
test = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_test/"

model = DNN(train, test, i, "rhythm")
