#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts

from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.layers import batch_norm


class SequentialNNBN:

    nOutputs = 88

    def __init__(self, ngramsLen, data, mode, encoder, inputSize):
        self.ngramsLen = ngramsLen
        shape = inputSize * (ngramsLen - 1)
        self.X = tf.placeholder(tf.float32, shape=(None,shape), name="X")
        self.y = tf.placeholder(tf.int32, name='y')
        self.is_training = tf.placeholder(tf.bool, name = "is_training")

        self.trainX, self.testX, self.trainY, self.testY = utils.setup_ngrams(data,
                                                        self.ngramsLen, mode, encoder)



    '''
    Returns next batch of specified size.
    Inputs:
        size = batch size
        X = input matrix
        y = targets vector
        iteration = batch iteration
    '''
    def next_batch(self, size, X, y, iteration):
        start = size * iteration
        stop = start + size
        return X[start:stop], y[start:stop]


    def construct(self, architecture, act, lossFun, keep_prob, learningRate = 0.01):

        bn_params = {"is_training": self.is_training, "decay": 0.999, 
            "updates_collections": None, "scale": True}

        with tf.name_scope('network'):

            z = fc(self.X, (architecture[0]), act, normalizer_fn = batch_norm, 
                normalizer_params = bn_params)

            z_drop = tf.contrib.layers.dropout(z, keep_prob,  is_training = self.is_training)

            for i in range(1, len(architecture)):

                name = 'layer' + str(i)
                z = fc(z_drop, (architecture[i]), act, normalizer_fn = batch_norm, 
                    normalizer_params = bn_params)

                z_drop = tf.contrib.layers.dropout(z, keep_prob,  is_training = self.is_training)

            self.logits = fc(z, SequentialNNBN.nOutputs, activation_fn = None, 
                normalizer_fn = batch_norm, normalizer_params = {"is_training": self.is_training, 						"decay": 0.999, "updates_collections": None, "scale": False})
   
        # error
        with tf.name_scope('loss'):
            error = lossFun(labels=self.y, logits=self.logits)
            loss = tf.reduce_mean(error, name='loss')

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learningRate)
            self.trainOp = optimizer.minimize(loss)

        # metrics used to evaluate network added here
        with tf.name_scope('evaluation'):
            # accuracy
            #@TODO: use softmax instead of logits if other metrics besides
            # correctness are used
            topK = tf.nn.in_top_k(self.logits, self.y, 1) # array of batchSize of bools
            self.accuracy = tf.reduce_mean(tf.cast(topK, tf.float32))
            lp = lossFun(labels=self.y, logits=self.logits)
            self.log_prob = tf.reduce_mean(lp, name='log_prob')


    def execute(self, batchSize, nEpochs):

        init = tf.global_variables_initializer()
        nBatches = int(np.ceil(len(self.trainX) / batchSize))

        best_lpTest = 10000
        best_state = None

        with tf.Session() as s:
            init.run()

            for epoch in range(nEpochs):
                # randomizes batches
                Xshuffle, Yshuffle = shuffle(self.trainX, self.trainY)

                for iteration in range(nBatches):
                    Xbatch, Ybatch = self.next_batch(batchSize, Xshuffle,
                                                    Yshuffle, iteration)
                    # feed_dict gives values to placeholders
                    s.run(self.trainOp, feed_dict={self.X: Xbatch, self.y: Ybatch,
                                                        self.is_training: True})


                train_dict = {self.X:Xshuffle, self.y:Yshuffle, self.is_training : False}
                test_dict = {self.X:self.testX, self.y:self.testY, self.is_training : False}


                accTrain = self.accuracy.eval(feed_dict=train_dict)
                accTest = self.accuracy.eval(feed_dict=test_dict)

                lpTrain = self.log_prob.eval(feed_dict=train_dict)
                lpTest = self.log_prob.eval(feed_dict=test_dict)
                if(lpTest < best_lpTest):
                    best_lpTest = lpTest
                    best_state = str(epoch) + " Train log prob: " + str(lpTrain) + " Test log prob: " +  str(lpTest)

                print(epoch, "Train accuracy:", accTrain, "Test accuracy:", accTest)
                print(epoch, "Train log prob:", lpTrain, "Test log prob:", lpTest)

            print("\n\nBest_state", best_state)


