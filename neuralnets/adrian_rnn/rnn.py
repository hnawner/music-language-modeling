#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import rnn_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts
from tensorflow.contrib.layers import fully_connected

class RNN:

    n_outputs = 88
    n_inputs = 20

    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.X_train, self.X_test, self.y_train, self.y_test = rnn_utils.setup_rnn(data_folder)
        self.seq_length = rnn_utils.max_length(self.X_train)


    def construct(self, architecture, cell_type, learningRate = 0.001):
        self.arch = architecture
        self.cell_type = cell_type

        # placeholders
        self.X = tf.placeholder(tf.float32, shape=[None, self.seq_length, RNN.n_inputs], name="X")
        self.y = tf.placeholder(tf.float32, shape=[None, self.seq_length, RNN.n_outputs], name='y')
        self.seq_lens = tf.placeholder(tf.int32, [None])


        cells = []

        for index in range(len(architecture)):
            cell = cell_type(num_units = architecture[index], use_peepholes = True)
            cells.append(cell)
        
        # build network
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(multi_layer_cell, self.X, dtype = tf.float32, sequence_length = self.seq_lens, swap_memory = True)

        # reshape outputs
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, architecture[(len(architecture) - 1)]])
        stacked_outputs = tf.contrib.layers.fully_connected(stacked_rnn_outputs, RNN.n_outputs, activation_fn = None)
        self.outputs = tf.reshape(stacked_outputs, [-1, self.seq_length, RNN.n_outputs])


        loss = self.loss_fn(self.outputs, self.y)

        optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
        self.trainOp = optimizer.minimize(loss)

        self.log_prob = self.loss_fn(self.outputs, self.y)
        self.accuracy = self.accuracy_fn(self.outputs, self.y)


    def execute(self, nEpochs, batch_size):
        nEpochs = nEpochs
        batch_size = batch_size
        nBatches = int(np.ceil(len(self.X_train) / batch_size))


        init = tf.global_variables_initializer()

        self.print_details()


        with tf.Session() as s:
            init.run()
            for epoch in range(nEpochs):

                Xshuffle, Yshuffle = shuffle(self.X_train, self.y_train)

                for iteration in range(nBatches):
                    Xbatch, Ybatch = self.next_batch(batch_size, Xshuffle, Yshuffle, iteration)
                    lengths = rnn_utils.get_mel_lengths(Xbatch)

                    s.run(self.trainOp, feed_dict={self.X: Xbatch, self.y: Ybatch, self.seq_lens: lengths})

                lpTrain = self.log_prob.eval(feed_dict={self.X : Xshuffle, self.y : Yshuffle, self.seq_lens:rnn_utils.get_mel_lengths(Xshuffle)})
                lpTest = self.log_prob.eval(feed_dict={self.X : self.X_test, self.y : self.y_test, self.seq_lens: rnn_utils.get_mel_lengths(self.X_test)})

                accTrain = self.accuracy.eval(feed_dict={self.X : Xshuffle, self.y : Yshuffle, self.seq_lens: rnn_utils.get_mel_lengths(Xshuffle)})
                accTest = self.accuracy.eval(feed_dict={self.X : self.X_test, self.y : self.y_test, self.seq_lens: rnn_utils.get_mel_lengths(self.X_test)})

                print(epoch, "Train log prob:", lpTrain, "Test log prob:", lpTest)
                print(epoch, "Train accuracy:", accTrain, "Test accuracy:", accTest)


    def print_details(self):
        print("Architecture:", self.arch)
        print("\nCell type:", self.cell_type)
        print("\nMels: ", self.data_folder)



    def loss_fn(self, out, y):
        cross_ent = y * tf.log(tf.nn.softmax(out))
        cross_ent = -tf.reduce_sum(cross_ent, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
        cross_ent = cross_ent * mask 
        cross_ent = tf.reduce_sum(cross_ent, reduction_indices=1)
        cross_ent = cross_ent / tf.reduce_sum(mask, reduction_indices=1)
        return tf.reduce_mean(cross_ent)



    def next_batch(self, size, X, y, iteration):
        start = size * iteration
        stop = start + size
        return X[start:stop], y[start:stop]


    def accuracy_fn(self, out, y):
        out_vals, out_ind = tf.nn.top_k(out, 1)
        y_vals, y_ind = tf.nn.top_k(y, 1)
        compare = tf.equal(out_ind, y_ind)
        mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
        compare = tf.reduce_mean(tf.cast(compare, tf.float32), axis = 2)
        acc = compare * mask 
        acc = tf.reduce_sum(acc, reduction_indices = 1)
        acc = acc / tf.reduce_sum(mask, reduction_indices=1)
        return tf.reduce_mean(acc)







