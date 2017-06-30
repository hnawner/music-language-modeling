#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from utils import setup_ngrams, one_hot_ngram_CNN

class CNN:

    nOutputs = 88

    def __init__(self, ngramLen, path):

        self.ngramLen = ngramLen
        self.train_x, self.test_x, self.train_y, self.test_y = setup_ngrams(
                path, ngramLen, one_hot_ngram_CNN)
        self.trainLen, self.height, self.nWidth, self.nChan = np.shape(self.train_x)
        self.X = tf.placeholder(tf.float32, shape=[None, self.height,
                    self.nWidth, self.nChan], name="X")
        self.Y = tf.placeholder(tf.int64, shape=[None], name="Y")


    def conv_layers(filters):
        conv = tf.nn.conv2d(self.X, filters[0], [1,1], 'SAME')
        filters.pop(0)
        while filters != []:
            z = tf.nn.conv2d(conv, filters[0], [1,1], "SAME")
            conv = z
            filters.pop(0)
        return conv

    def construct(self):
        network = conv_layers(self.filters)

        dims = network.get_shape().as_list()
        new_dim = dims[1] * dims[2] * dims[3]
        network_flat = tf.reshape(tensor=network, shape=[-1, new_dim])

        logits = tf.contrib.layers.fully_connected(network_flat, CNN.nOutputs,
                                                activation_fn=None)

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y,
                                                                  logits=logits)
        self.loss = tf.reduce_mean(xentropy, name='loss')
        self.train_op = (tf.train.GradientDescentOptimizer(self.learn).minimize(
                                                                    self.loss))

        correct = tf.nn.in_top_k(logits, self.Y, 1)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def next_batch(self, size, X, y, iteration):
        start = size * iteration
        stop = start + size
        return X[start:stop], y[start:stop]

    def execute(self, batch_size, n_epochs):
        n_batches = int(np.ceil(len(self.train_x / batch_size)))

        with tf.Session() as s:
            tf.global_variables_initializer().run()
            for epoch in range(n_epochs):
                x_shuffle, y_shuffle = shuffle(self.train_x, self.train_y)
                for iteration in range(n_batches):
                    x_batch, y_batch = self.next_batch(batch_size, x_shuffle,
                                                    y_shuffle, iteration)
                    s.run(self.train_op, feed_dict={self.X: x_batch,
                                                    self.Y: y_batch})
                log_tr = self.loss.eval(feed_dict={self.X: self.train_x,
                                                    self.Y: self.train_y})
                acc_tr = self.accuracy.eval(feed_dict={self.X: self.train_x,
                                                    self.Y: self.train_y})
                log_te = self.loss.eval(feed_dict={self.X: self.test_x,
                                                    self.Y: self.test_y})
                acc_te = self.accuracy.eval(feed_dict={self.X: self.test_x,
                                                    self.Y: self.test_y})

                print('--- ', epoch, ' ---')
                print("LP -tr", log_tr, "-te", log_te)
                print("ACC -tr", acc_tr, "-te", acc_te, '\n')
                
