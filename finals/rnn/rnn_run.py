#!/usr/bin/env python

from rnn import RNN

data_tr = '/home/hawner2/reu/musical-forms/mels/pitches/folk_major'
data_te = '/home/hawner2/reu/musical-forms/mels/pitches/folk_maj_test'

RNN(data_tr, data_te, 128)
# RNN(data_tr, data_te, 512)
