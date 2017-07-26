#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from utils import setup
from sklearn.utils import shuffle


class RNN:

    min_pitch = 40
    max_pitch = 95
    n_outputs = max_pitch - min_pitch
    n_features = 20
    batch_size = 64
    n_epochs = 25

    def __init__(self, data_tr, data_te, n_units):
        self.tr_x, self.tr_y, self.te_x, self.te_y, self.max_len = setup(
                                                                   data_tr,
                                                                   data_te,
                                                                   RNN.min_pitch,
                                                                   RNN.n_outputs)
        self.X = tf.placeholder(tf.float32,
                                shape=[None, self.max_len, RNN.n_features],
                                name="X")
        self.y = tf.placeholder(tf.float32,
                                shape=[None, self.max_len, RNN.n_outputs],
                                name='y')
        self.n_units = n_units
        self.build()

    def loss_fn(self):
        xentropy = self.y * tf.log(tf.nn.softmax(self.logits))
        xentropy = -tf.reduce_sum(xentropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.y), reduction_indices=2))
        xentropy *= mask
        xentropy = tf.reduce_sum(xentropy, reduction_indices=1)
        xentropy /= tf.reduce_sum(mask, reduction_indices=1)
        return tf.reduce_mean(xentropy)

    def accuracy_fn(self):
        _, preds = tf.nn.top_k(self.logits, 1)
        _, true = tf.nn.top_k(self.y, 1)
        correct = tf.equal(preds, true)
        mask = tf.sign(tf.reduce_max(tf.abs(self.y), reduction_indices=2))
        acc = tf.reduce_mean(tf.cast(correct, tf.float32), axis=2)
        acc *= mask
        acc = tf.reduce_sum(acc, reduction_indices=1)
        acc /= tf.reduce_sum(mask, reduction_indices=1)
        return tf.reduce_mean(acc)

    def build(self):

        with tf.name_scope('rnn'):
            cells = []
            for _ in range(3):
                c = tf.nn.rnn_cell.LSTMCell(self.n_units, use_peepholes=True,
                                            activation=tf.tanh)
                cells.append(c)

            multi = tf.nn.rnn_cell.MultiRNNCell(cells)
            outs, _ = tf.nn.dynamic_rnn(multi, self.X, dtype=tf.float32,
                                            swap_memory=True)

            outs = tf.reshape(outs, [-1, self.n_units])
            logits = tf.layers.dense(outs, RNN.n_outputs)
            self.logits = tf.reshape(logits, [-1, self.max_len, RNN.n_outputs])

        with tf.name_scope('optimizer'):
            loss = self.loss_fn()
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            self.train_op = optimizer.minimize(loss)

        with tf.name_scope('evaluation'):
            self.lp = loss
            self.accuracy = self.accuracy_fn()

        self.init = tf.global_variables_initializer()
        self.execute()

    def next_batch(self, x, y, batch):
        start = RNN.batch_size * batch
        stop = start + RNN.batch_size
        return x[start:stop], y[start:stop]

    def execute(self):
        n_batches = int(np.ceil(len(self.tr_x) / RNN.batch_size))

        with tf.Session() as s:
            self.init.run()

            print('--------------')
            print('RNN, units %d' % self.n_units)
            print('--------------')

            for e in range(self.n_epochs):
                shuf_x, shuf_y = shuffle(self.tr_x, self.tr_y)

                for b in range(n_batches):
                    xbatch, ybatch = self.next_batch(shuf_x, shuf_y, b)
                    feed_dict = {self.X: xbatch, self.y: ybatch}
                    s.run(self.train_op, feed_dict=feed_dict)

                feed_dict = {self.X: self.tr_x, self.y: self.tr_y}
                acc = self.accuracy.eval(feed_dict=feed_dict)
                lp = self.lp.eval(feed_dict=feed_dict)

                print("%d   acc: %.3f   lp: %.3f" % (e, acc, lp))

            ### final validation tests ###
            n_batches_te = int(np.ceil(len(self.te_x) / RNN.batch_size))
            acc_te = 0
            lp_te = 0

            for b in range(n_batches_te):
                xbatch, ybatch = self.next_batch(self.te_x, self.te_y, b)
                fd_te = {self.X: xbatch, self.y: ybatch}
                acc_te += self.accuracy.eval(feed_dict=fd_te)
                lp_te += self.lp.eval(feed_dict=fd_te)

            print('accuracy: %.3f' % (acc_te / n_batches_te))
            print('log prob: %.3f' % (lp_te / n_batches_te))


