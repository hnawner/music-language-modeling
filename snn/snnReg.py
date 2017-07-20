#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts


class SequentialNNReg:

    nOutputs = 88

    def __init__(self, details, ngramsLen, data, mode, encoder, inputSize):
        self.details = details
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


    def construct(self, architecture, act, lossFun, regFun, reg_scale, learningRate = 0.01):

        with tf.name_scope("network"):


             
            reg = regFun(scale = reg_scale)

            z = tf.contrib.layers.fully_connected(self.X, num_outputs = architecture[0], activation_fn = act, weights_regularizer = reg)

            for i in range(1, len(architecture)):

                name = 'layer' + str(i)
                z = tf.contrib.layers.fully_connected(z, architecture[i], activation_fn = act, weights_regularizer = reg)

            self.logits = tf.contrib.layers.fully_connected(z, SequentialNNReg.nOutputs, activation_fn = None, weights_regularizer = reg)

        #self.get_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.get_weights = tf.trainable_variables()
        #self.saver = tf.train.Saver()

   
        # error
        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            error = lossFun(labels=self.y, logits=self.logits)
            self.reg_loss_val = tf.reduce_sum(reg_loss)
            loss = tf.reduce_mean(error) + self.reg_loss_val

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdagradOptimizer(learningRate)
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


    def execute(self, batchSize, nEpochs, filename):
        f = open(filename, "a+")
        best_lpTest = 10000
        best_state = None

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

                lpTrain = self.log_prob.eval(feed_dict={self.X:Xshuffle,
                                                            self.y:Yshuffle})
                lpTest = self.log_prob.eval(feed_dict={self.X:self.testX,
                                                            self.y:self.testY})
                regLoss = self.reg_loss_val.eval(feed_dict={self.X:Xshuffle,
                                                            self.y:Yshuffle})
                

                if(lpTest < best_lpTest):
                    best_lpTest = lpTest
                    best_state = "\n" + str(epoch) + " Train log prob: " + str(lpTrain) + " Test log prob: " +  str(lpTest) + "\n\n"

                print(epoch, "Train accuracy:", accTrain, "Test accuracy:", accTest)
                print(epoch, "Train log prob:", lpTrain, "Test log prob:", lpTest)
                print(epoch, "Regularization loss:", regLoss)

            #self.saver.save(s, "session.tf")

            print("Best_state", best_state)

            weights = s.run(self.get_weights)
            print(len(weights))
            for i in range(len(weights)):
                np.savetxt((str(i) + "weights.txt"), weights[i], delimiter = "\t", fmt = "%.8f")
        f.write(self.details)
        f.write(best_state)


