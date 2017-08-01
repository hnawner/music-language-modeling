#!/usr/bin/env python

from rnn import RNN

train = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_train/"
test = "/home/aeldrid2/REU/krn_split/converted/maj_min_split/maj_test/"
n_neurons = 256
data_type = "pitch"

model = RNN(train, test, n_neurons, data_type)
