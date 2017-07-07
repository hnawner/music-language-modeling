#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import rhythm_utils as utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts
from note2vec import note2vec, build_dataset


class SNN:

    nInputs = 20

    def __init__(self, ngramsLen, master_folder, data, data_type):
        self.ngramsLen = ngramsLen
        shape = (ngramsLen - 1)
        self.X = tf.placeholder(tf.int32, shape=(None,shape), name="X")
        self.y = tf.placeholder(tf.int32, name='y')
        self.perform_embed(master_folder, data, ngramsLen)

    def perform_embed(self, master_folder, folder, ngramsLen):
        r_dict = utils.build_rhythm_dict(master_folder)
        data = utils.read_files_to_embed(folder, r_dict)
        train, test = tts(data, test_size = 0.2)
        dictionary = note2vec(train, r_dict, "embed_dict.txt")
        self.n_out = len(dictionary)

        embed_train = []
        embed_test = []       
        for train_mel in train:
            embed_train_mel = []
            for train_note in train_mel:
                embed_train_mel.append(dictionary[train_note])
            embed_train.append(embed_train_mel)

        for test_mel in test:
            embed_test_mel = []
            for test_note in test_mel:
                if test_note in dictionary:
                    embed_test_mel.append(dictionary[test_note])
                else: embed_test_mel.append(dictionary[-1])
            embed_test.append(embed_test_mel)

        train_grams = utils.make_ngrams(embed_train, ngramsLen)
        test_grams = utils.make_ngrams(embed_test, ngramsLen)

        self.trainX, self.trainY = utils.split_inst_targets(train_grams)
        self.testX, self.testY = utils.split_inst_targets(test_grams)
         
        
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

        # embed data
        embeddings = np.genfromtxt("embed_dict.txt")
        inputs = tf.nn.embedding_lookup(embeddings, self.X)
        inputs = tf.reshape(inputs, [-1, SNN.nInputs * (self.ngramsLen - 1)])
        inputs = tf.cast(inputs, tf.float32)


        with tf.name_scope('network'):

            z = self.new_layer(inputs, (architecture[0]), 'layer0', act)

            for i in range(1, len(architecture)):

                name = 'layer' + str(i)
                z = self.new_layer(z, (architecture[i]), name, act)

            self.logits = self.new_layer(z, self.n_out, 'out')
   
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
            topK = tf.nn.in_top_k(self.logits, self.y, 1)
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
                    s.run(self.trainOp, feed_dict={self.X: Xbatch,
						 self.y: Ybatch})

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
                    best_state = str(epoch) + " Train log prob: " + str(lpTrain) 
						+ " Test log prob: " +  str(lpTest)

                print(epoch, "Train accuracy:", accTrain,
					 "Test accuracy:", accTest)
                print(epoch, "Train log prob:", lpTrain, 
					"Test log prob:", lpTest)

            print("\n\nBest_state", best_state)


