#!/usr/bin/env python
# feedforward.py

'''
This is a simple, multi-classification, feed-forward neural network that,
when given some number of preceding pitches, can predict the next pitch in
the sequence (not taking into account rhythm).
'''

from __future__ import division, print_function

import utils
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda

# -------- Construction -------- #

# Parameters
pitch_range = 88
ngram_len = 2
batch_size = 32
epochs = 2

input_size = (ngram_len - 1) * pitch_range

(X_train, y_train), (X_test, y_test) = utils.setup()

model = Sequential()
# input layer
model.add(Dense(input_size,
                input_dim=pitch_range,
                input_length=ngram_len,
                activaton='relu'))
# output layer
model.add(Dense(pitch_range, activation='sigmoid'))

# formatting layer
def vector_to_index(X):
    maxind = -1
    for i in range(len(X)):
        if X[i] > X[mamxind]:
            maxind = i
    return maxind

model.add(Lambda(vector_to_index, output_shape(1,)))

# -------- Execution -------- #

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))


