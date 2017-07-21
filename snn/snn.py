#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts


class SequentialNN:

    nOutputs = 88

    def __init__(self, ngramsLen, data, mode, encoder, inputSize):
        self.ngramsLen = ngramsLen
        shape = inputSize * (ngramsLen - 1)
        self.X = tf.placeholder(tf.float32, shape=(None,shape), name="X")
        self.y = tf.placeholder(tf.int32, name='y')
        #self.pitchRange = inputSize
        self.trainX, self.testX, self.trainY, self.testY = utils.setup_ngrams(data,
                                                        self.ngramsLen, mode, encoder)

        
    '''
    Creates a new neuron layer and returns activation.
    Inputs:
        X = input matrix
        n = number of neurons
        name = name scope of layer
        g = activation function applied to weighted sum
    Outputs:
        g ( WX + b )
        g = activation function
        W = weight matrix
        X = input matrix
        b = bias
    '''
    def new_layer(self, X, n, name = 'layer', g = None):
        #@TODO: explanation of steps
        with tf.name_scope(name):
            nInputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(nInputs)

            init = tf.truncated_normal((nInputs, n), stddev=stddev)

            W = tf.Variable(init, name='weights')
            b = tf.Variable(tf.zeros([n]), name='biases')
        
            a = tf.matmul(X, W) + b
            z = g(a) if g != None else a

            return z


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


    def construct(self, architecture, act, lossFun, learningRate = 0.01):

        with tf.name_scope('network'):

            z = self.new_layer(self.X, (architecture[0]), 'layer0', act)

            for i in range(1, len(architecture)):

                name = 'layer' + str(i)
                z = self.new_layer(z, (architecture[i]), name, act)

            self.logits = self.new_layer(z, SequentialNN.nOutputs, 'out')
   
        # error
        with tf.name_scope('loss'):
            error = lossFun(labels=self.y, logits=self.logits)
            loss = tf.reduce_mean(error, name='loss')

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer()
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
                    s.run(self.trainOp, feed_dict={self.X: Xbatch, self.y: Ybatch})

                    # evaluates network with entire set of values
                    # with current weights
                accTrain = self.accuracy.eval(feed_dict={self.X:Xshuffle,
                                                            self.y:Yshuffle})
                accTest = self.accuracy.eval(feed_dict={self.X:self.testX,
                                                            self.y:self.testY})

                lpTrain = self.log_prob.eval(feed_dict={self.X:Xshuffle,
                                                            self.y:Yshuffle})
                lpTest = self.log_prob.eval(feed_dict={self.X:self.testX,
                                                            self.y:self.testY})
                if(lpTest < best_lpTest):
                    best_lpTest = lpTest
                    best_state = str(epoch) + " Train log prob: " + str(lpTrain) + " Test log prob: " +  str(lpTest)

                print(epoch, "Train accuracy:", accTrain, "Test accuracy:", accTest)
                print(epoch, "Train log prob:", lpTrain, "Test log prob:", lpTest)

            print("\n\nBest_state", best_state)


