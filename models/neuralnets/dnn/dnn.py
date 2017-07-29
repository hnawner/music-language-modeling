#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import utils
import sys
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts


class DNN:

    min_pitch = 40
    max_pitch = 95
    n_outputs = (max_pitch - min_pitch)
    ngram = None
    n_features = None
    batch_size = 64
    n_epochs = 250

    def __init__(self, data_tr, data_te, n, d_type, model_type = 'default', trans = True):
        DNN.ngram = n
        self.tr_x, self.tr_y, self.te_x, self.te_y = utils.setup(data_tr,
                                                           data_te, 
                                                           DNN.ngram,
                                                           d_type,
                                                           DNN.min_pitch,
                                                           DNN.n_outputs,
                                                           trans)
        self.tr_x, self.val_x, self.tr_y, self.val_y = tts(self.tr_x, self.tr_y,
        	test_size = 0.2)
        if d_type == "pitch":
        	DNN.n_outputs = DNN.max_pitch - DNN.min_pitch
        	DNN.n_features = 20 * (DNN.ngram - 1)
        elif d_type == "rhythm":
        	DNN.n_outputs = len(utils.r_dict)
        	DNN.n_features = len(utils.r_dict) * (DNN.ngram - 1)   	
        self.X = tf.placeholder(tf.float32, shape=[None, DNN.n_features], name="X")
        self.y = tf.placeholder(tf.int32, shape=[None], name='y')
        if model_type == 'bn':
            self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.model = model_type
        self.save_path = "dnn"+d_type+model_type+str(trans)+str(DNN.ngram)+"/best.ckpt"

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
            if self.model == 'bn':
                fc1 = tf.contrib.layers.dropout(fc1, 0.6,
                                                is_training=self.is_training)
            fc2 = tf.contrib.layers.fully_connected(fc1, 512,
                                    activation_fn=tf.tanh,
                                    normalizer_fn=bn,
                                    normalizer_params=bn_params,
                                    weights_regularizer=reg)            
            if self.model == 'bn':
                fc2 = tf.contrib.layers.dropout(fc2, 0.6,
                                                is_training=self.is_training)
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
                                    DNN.n_outputs,
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
        self.saver = tf.train.Saver()

        self.execute()

    def next_batch(self, x, y, batch):
        start = DNN.batch_size * batch
        stop = start + DNN.batch_size
        return x[start:stop], y[start:stop]

    def execute(self):
        n_batches = int(np.ceil(len(self.tr_x) / DNN.batch_size))

        with tf.Session() as s:
            self.init.run()
            
            best = sys.maxint

            print('----------------------')
            print('DNN model type', self.model)
            print('----------------------')

            for epoch in range(DNN.n_epochs):
                shuf_x, shuf_y = shuffle(self.tr_x, self.tr_y)

                for batch in range(n_batches):
                    Xbatch, Ybatch = self.next_batch(shuf_x, shuf_y, batch)
                    feed_dict = {self.X: Xbatch, self.y: Ybatch}
                    if self.model == 'bn':
                        feed_dict[self.is_training] = True
                    s.run(self.train_op, feed_dict=feed_dict)

                feed_dict = {self.X: self.tr_x, self.y: self.tr_y}
                if self.model == 'bn':
                    feed_dict[self.is_training] = False
                acc = self.accuracy.eval(feed_dict=feed_dict)
                lp = self.lp.eval(feed_dict=feed_dict)
                
                feed_dict = {self.X: self.val_x, self.y: self.val_y}
                if self.model == 'bn':
                    feed_dict[self.is_training] = False                
                val_acc = self.accuracy.eval(feed_dict=feed_dict)
                val_lp = self.lp.eval(feed_dict=feed_dict)

                print("train %d   acc %.3f    lp %.3f" % (epoch, acc, lp))
                print("tune %d   acc %.3f    lp %.3f" % (epoch, val_acc, val_lp))

                
                if val_lp < best:
                    best = val_lp
                    self.saver.save(s, self.save_path)

        ### final validation tests ###
        with tf.Session() as s:
            self.saver.restore(s, self.save_path)
            feed_dict = {self.X: self.te_x, self.y: self.te_y}
            if self.model == 'bn':
                feed_dict[self.is_training] = False

            acc = self.accuracy.eval(feed_dict=feed_dict)
            lp = self.lp.eval(feed_dict=feed_dict)

            print("accuracy: %.3f" % acc)
            print("log prob: %.3f" % lp)
