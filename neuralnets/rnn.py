#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tf.contrib.layers import fully_connected
from tf.nn import sparse_softmax_cross_entropy_with_logits as ssxe
from sklearn.utils import shuffle

from utils import setupRNN

class RNN:

    def __init__(self, path, mode = 'major'):

        self.X = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        self.trainx, self.testx, self.trainy, self.testy = setupRNN(path, mode)


    def next_batch(self, size, X, y, iteration):
        start = size * iteration
        stop = start + size
        return X[start:stop], y[start:stop]

    def construct(self, n_neurons, activation = tf.tanh, loss_fun, mu = 0.001):

        n_outputs = 88

        cell = tf.contrib.nn.BasicRNNCell(num_units=n_neurons)
        outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

        # collapses output to correct number of neurons
        logits = fully_connected(states, n_outputs, activation_fn=None)

        # optimize
        xentropy = ssxe(labnels=self.y, logits=logits)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=mu)
        train_op = optimizer.minimize(loss) # operation used in execution

        # metrics
        correct = tf.nn.in_top_k(logits, self.y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    def execute(self, batch_size, n_epochs):

        init = tf.global_variables_initializer()

        n_batches = len(self.trainx) // batch_size

        with tf.Session() as s:
            init.run()

            for epoch in range(n_epochs):
                shufflex, shuffley = shuffle(self.trainx, self.trainy)

                for iteration in range(n_batches):
                    xbatch, ybatch = next_batch(batch_size, shufflex,
                                                shuffley, iteration)
                    s.run(self.train_op, feed_dict={X:xbatch, y:ybatch})

                acc_train = accuracy.eval(feed_dict={X:shufflex, y:shuffley})
                acc_test = accuracy.eval(feed_dict={X:self.testx, y:self.testy})
                print(epoch, "Train accuracy:", acc_train,
                             "Test accuracy:", acc_test)


