#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
from utils import setup

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts


class SNN:

    min_pitch = 40
    max_pitch = 95
    n_outputs = max_pitch - min_pitch
    ngram = 8
    n_features = 20 * (ngram - 1)
    batch_size = 64
    n_epochs = 100

    def __init__(self, data_tr, data_te, model_type = 'default'):
        self.tr_x, self.tr_y, self.te_x, self.te_y = setup(data_tr,
                                                           data_te,
                                                           SNN.ngram,
                                                           SNN.min_pitch,
                                                           SNN.n_outputs)
        self.X = tf.placeholder(tf.float32, shape=[None, SNN.n_features], name="X")
        self.y = tf.placeholder(tf.int32, shape=[None], name='y')
        if model_type == 'bn':
            self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.model = model_type

        self.build()

    def build(self):
        bn = None
        bn_params = None
        reg = None

        if self.model == 'l2':
            reg = tf.contrib.layers.l2_regularizer(scale = 0.0000004)
        elif self.model == 'bn':
            bn = tf.contrib.layers.batch_norm
            bn_params = {'is_training': self.is_training, 'decay': 0.999,
                        'updates_collections': None, 'scale': True}
        
        with tf.name_scope('network'):

            fc1 = tf.contrib.layers.fully_connected(self.X, 512,
                                    activation_fn=tf.tanh,
                                    normalizer_fn=bn,
                                    normalizer_params=bn_params,
                                    weights_regularizer=reg)
            fc2 = tf.contrib.layers.fully_connected(fc1, 512,
                                    activation_fn=tf.tanh,
                                    normalizer_fn=bn,
                                    normalizer_params=bn_params,
                                    weights_regularizer=reg)
            fc3 = tf.contrib.layers.fully_connected(fc2, 512,
                                    activation_fn=tf.tanh,
                                    normalizer_fn=bn,
                                    normalizer_params=bn_params,
                                    weights_regularizer=reg)
            if self.model == 'bn':
                fc3 = tf.contrib.layers.dropout(fc3, 0.6,
                                                is_training=self.is_training)
                bn_params['scale'] = False
            self.logits = tf.contrib.layers.fully_connected(fc3,
                                    SNN.n_outputs,
                                    activation_fn=None,
                                    normalizer_fn=bn,
                                    normalizer_params=bn_params,
                                    weights_regularizer=reg)

        with tf.name_scope('optimizer'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                        labels=self.y,
                                                        logits=self.logits)
            loss = tf.reduce_mean(loss)
            if self.model == 'l2':
                reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                loss += tf.reduce_sum(reg_loss)

            optimizer = tf.train.GradientDescentOptimizer(0.005)
            self.train_op = optimizer.minimize(loss)

        with tf.name_scope('evaluation'):
            # log likelihood
            lp = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                        labels=self.y,
                                                        logits=self.logits)
            self.lp = tf.reduce_mean(lp)

            # accuracy
            correct = tf.nn.in_top_k(self.logits, self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.init = tf.global_variables_initializer()

        self.execute()

    def next_batch(self, x, y, batch):
        start = SNN.batch_size * batch
        stop = start + SNN.batch_size
        return x[start:stop], y[start:stop]

    def execute(self):
        n_batches = int(np.ceil(len(self.tr_x) / SNN.batch_size))

        with tf.Session() as s:
            self.init.run()

            print('----------------------')
            print('SNN model type', self.model)
            print('----------------------')

            for epoch in range(SNN.n_epochs):
                shuf_x, shuf_y = shuffle(self.tr_x, self.tr_y)

                for batch in range(n_batches):
                    Xbatch, Ybatch = self.next_batch(shuf_x, shuf_y, batch)
                    feed_dict = {self.X: Xbatch, self.y: Ybatch}
                    if self.model == 'bn':
                        feed_dict[self.is_training] = True
                    s.run(self.train_op, feed_dict=feed_dict)

                feed_dict = {self.X: self.tr_x, self.y: self.tr_y}
                acc = self.accuracy.eval(feed_dict=feed_dict)
                lp = self.lp.eval(feed_dict=feed_dict)

                print("%d   acc %.3f    lp %.3f" % (epoch, acc, lp))

            ### final validation tests ###
            feed_dict = {self.X: self.te_x, self.y: self.te_y}
            if self.model == 'bn':
                feed_dict[self.is_training] = False

            acc = self.accuracy.eval(feed_dict=feed_dict)
            lp = self.lp.eval(feed_dict=feed_dict)

            print("accuracy: %.3f" % acc)
            print("log prob: %.3f" % lp)
