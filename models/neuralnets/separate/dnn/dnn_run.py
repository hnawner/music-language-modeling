#!/usr/bin/env python

from dnn import DNN

train = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_train/"
test = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_test/"
n = 8
data_type = "rhythm"
model_type = "default"
transpose = True
encode = "pc"

model = DNN(train, test, n, data_type, model_type, transpose, encode)
