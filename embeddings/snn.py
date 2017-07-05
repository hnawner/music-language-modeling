#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts


class SequentialNN:

    pitchRange = 88
    nOutputs = 88

    def __init__(self, ngramsLen, data):
        self.ngramsLen = ngramsLen
        shape = SequentialNN.pitchRange * (ngramsLen - 1)
        self.X = tf.placeholder(tf.float32, shape=(None,shape), name="X")
        self.y = tf.placeholder(tf.int32, name='y')

        self.trainX, self.testX, self.trainY, self.testY = utils.setupSNN(data,
                                                        self.ngramsLen)

        self.embeddings = np.genfromtxt('embedding_dict.txt')

        

        
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


    def construct(self, architecture):

        tf.nn.embedding_lookup(self.embeddings, self.X)

        with tf.name_scope('network'):

            z = self.new_layer(self.X, (architecture[0]), 'layer0', tf.nn.relu)

            for i in range(1, len(architecture)):

                name = 'layer' + str(i)
                z = self.new_layer(z, (architecture[i]), name, act)

            logits = self.new_layer(z, SequentialNN.nOutputs, 'out')
   
        # error
        with tf.name_scope('loss'):
            error = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
            loss = tf.reduce_mean(error, name='loss')

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.trainOp = optimizer.minimize(loss)

        # metrics used to evaluate network added here
        with tf.name_scope('evaluation'):
            # accuracy
            #@TODO: use softmax instead of logits if other metrics besides
            # correctness are used
            topK = tf.nn.in_top_k(logits, self.y, 1) # array of batchSize of bools
            self.accuracy = tf.reduce_mean(tf.cast(topK, tf.float32))


    def execute(self, batchSize, nEpochs):

        init = tf.global_variables_initializer()
        nBatches = int(np.ceil(len(self.trainX) / batchSize))

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

                print(epoch, "Train accuracy:", accTrain, "Test accuracy:", accTest)



s = SequentialNN(5, '/home/hawner2/reu/musical-forms/mels/folk_major')
s.construct([128])
s.execute(32, 100)
