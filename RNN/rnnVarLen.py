#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts
from tensorflow.contrib.layers import fully_connected

# parameters
n_steps = 15
n_inputs = 20
n_neurons1 = 15
n_neurons2 = 14
n_outputs = 88
learn = 0.001

print("DATA INPUT \n\n")

# data input
maj_mels, min_mels = utils.read_files("/home/aeldrid2/REU/mels/folk_minor")
#records, labels = utils.make_rnn_data(min_mels, n_steps)
records, labels = utils.make_rnn_data_varlen(min_mels)

padded_mels = utils.pad(records)
padded_labels = utils.pad(labels)
seq_length = utils.get_mel_lengths(padded_mels)[0]
print(seq_length)
records = utils.rnn_encoder(padded_mels)
labels = utils.rnn_labels_encoder(padded_labels)
X_train, X_test, y_train, y_test = tts(records, labels, test_size = 0.2)
print(len(X_train))


print("BUILD NETWORK \n\n")

# build networks
X = tf.placeholder(tf.float32, shape=[None, seq_length, n_inputs], name="X")
y = tf.placeholder(tf.float32, shape=[None, seq_length, n_outputs], name='y')
seq_lens = tf.placeholder(tf.int32, [None])

cell1 = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons1, activation = tf.nn.relu)
cell2 = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons2, activation = tf.nn.relu)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype = tf.float32, sequence_length = seq_lens, swap_memory = True)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons2])
stacked_outputs = tf.contrib.layers.fully_connected(stacked_rnn_outputs, n_outputs, activation_fn = None)

outputs = tf.reshape(stacked_outputs, [-1, seq_length, n_outputs])


#y_vals, y_pred = tf.nn.top_k(outputs, 1)
#y_pred = tf.Print(y_pred, [y_pred], summarize = 100)


print("DEFINE LOSS \n\n")

def loss_fn(out, y):
    cross_ent = y * tf.nn.softmax(out)
    cross_ent = tf.reduce_sum(cross_ent, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
    cross_ent = cross_ent * mask 
    cross_ent = tf.reduce_sum(cross_ent, reduction_indices=1)
    cross_ent = cross_ent / tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(cross_ent)

loss = loss_fn(outputs, y)

optimizer = tf.train.AdamOptimizer(learning_rate = learn)
trainOp = optimizer.minimize(loss)



print("DEFINE EVAL \n\n")

#topK = tf.nn.in_top_k(tf.contrib.layers.flatten(tf.slice(outputs, [0,4,0], [tf.shape(outputs)[0], 1, 88])) , tf.reshape(tf.slice(y, [0,4], [tf.shape(y)[0], 1]), [-1]), 1)
#accuracy = tf.reduce_mean(tf.cast(topK, tf.float32))
#lp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = outputs)
log_prob = loss_fn(outputs, y)



def next_batch(size, X, y, iteration):
    start = size * iteration
    stop = start + size
    return X[start:stop], y[start:stop]

print("RUN SESSION \n\n")

nEpochs = 4
batch_size = 16

init = tf.global_variables_initializer()
nBatches = int(np.ceil(len(X_train) / batch_size))

with tf.Session() as s:
    init.run()
    for epoch in range(nEpochs):

        Xshuffle, Yshuffle = shuffle(X_train, y_train)
        Xs = np.asarray(Xshuffle)
        ys = np.asarray(Yshuffle)
        #print(Xs.shape)
        #print(ys.shape)

        for iteration in range(nBatches):
            Xbatch, Ybatch = next_batch(batch_size, Xshuffle, Yshuffle, iteration)
            lengths = utils.get_mel_lengths(Xbatch)
            s.run(trainOp, feed_dict={X: Xbatch, y: Ybatch, seq_lens: lengths})

        lpTrain = log_prob.eval(feed_dict={X : Xshuffle, y : Yshuffle, seq_lens: utils.get_mel_lengths(Xshuffle)})
        lpTest = log_prob.eval(feed_dict={X : X_test, y : y_test, seq_lens: utils.get_mel_lengths(X_test)})

        #accTrain = accuracy.eval(feed_dict={X : Xshuffle, y : Yshuffle})
        #accTest = accuracy.eval(feed_dict={X : X_test, y : y_test})

        print(epoch, "Train log prob:", lpTrain, "Test log prob:", lpTest)
        #print(epoch, "Train accuracy:", accTrain, "Test accuracy:", accTest)

    #X_new = [ [56, 67, 34, 56, 45], [24, 34, 67, 23, 90] ]
    #X_new = utils.rnn_encoder(X_new)

    #prediction = s.run(y_pred, feed_dict={X: X_new})


print("DONE \n\n")





