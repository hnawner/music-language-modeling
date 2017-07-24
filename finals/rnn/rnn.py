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
    n_epochs = 50

    def __init__(self, data_tr, data_te, n_units):
        self.tr_x, self.tr_y, self.te_x, self.te_y, self.max_len = setup(
                                                                   data_tr,
                                                                   data_te,
                                                                   RNN.min_pitch,
                                                                   RNN.n_outputs)
        self.n_units = n_units
        self.X = tf.placeholder(tf.float32,
                                shape=[None, self.max_len, RNN.n_features],
                                name="X")
        self.y = tf.placeholder(tf.float32,
                                shape=[None, self.max_len, RNN.n_outputs],
                                name='y')
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
                c = tf.nn.rnn_cell.LSTMCell(self.n_units,
                                            use_peepholes=True,
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
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = optimizer.minimize(loss)

        with tf.name_scope('evaluation'):
            self.lp = loss
            tf.summary.scalar('log likelihood', self.lp)
            self.accuracy = self.accuracy_fn()
            tf.summary.scalar('accuracy', self.accuracy)

        self.init = tf.global_variables_initializer()
        summaries = tf.summary.merge_all()
        #self.writer = tf.train.SummaryWriter('logs', tf.get_default_graph())
        self.execute()

    def next_batch(self, x, y, batch):
        start = RNN.batch_size * batch
        stop = start + RNN.batch_size
        return x[start:stop], y[start:stop]

    def execute(self):
        n_batches = int(np.ceil(len(self.tr_x) / RNN.batch_size))

        with tf.Session() as s:
            self.init.run()

            print('------------------')
            print('RNN, units %d' % self.n_units)
            print('------------------')

            for e in range(self.n_epochs):
                shuf_x, shuf_y = shuffle(self.tr_x, self.tr_y)

                for b in range(n_batches):
                    xbatch, ybatch = self.next_batch(shuf_x, shuf_y, b)
                    feed_dict = {self.X: xbatch, self.y: ybatch}
                    s.run(self.train_op, feed_dict=feed_dict)

            ### final validation tests ###
            fd_tr = {self.X: self.tr_x, self.y: self.tr_y}
            fd_te = {self.X: self.te_x, self.y: self.te_y}

            acc_tr = self.accuracy.eval(feed_dict=fd_tr)
            lp_tr = self.lp.eval(feed_dict=fd_tr)
            acc_te = self.accuracy.eval(feed_dict=fd_te)
            lp_te = self.lp.eval(feed_dict=fd_te)

            print('accuracy -tr: %.3f  -te: %.3f' % (acc_tr, acc_te))
            print('log prob -tr: %.3f  -te: %.3f' % (lp_tr, lp_te))


#RNN('/home/hawner2/reu/musical-forms/mels/pitches/train_all', '/home/hawner2/reu/musical-forms/mels/pitches/test_all', 128)
