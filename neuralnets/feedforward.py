#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts

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
def new_layer(X, n, name = 'layer', g = None):
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
def next_batch(size, X, y, iteration):
    start = size * iteration
    stop = start + size
    return X[start:stop], y[start:stop]

# -------------------- Construction  --------------------- #

ngramLen = 2
pitchRange = 88
nInputs = (ngramLen - 1) * pitchRange
nOutputs = pitchRange

nHidden1 = 256 # arbitrary

# placeholders cannot be directly eval'd; must be given input feed
# should these have shape parameters?
X = tf.placeholder(tf.int32, name='X')
y = tf.placeholder(tf.int32, name='y')

with tf.name_scope('network'):
    z1 = new_layer(X, nInputs, name='layer1')
    z2 = new_layer(z1, nHidden1, name='layer2')
    # unscaled final outputs go through softmax later
    logits = new_layer(z2, nOutputs, name='out')
   
# error
with tf.name_scope('loss'):
    # cross entropy used for multiclass classification where each
    # instance can only be part of one class
    # if this doesn't work, use:
    #   tf.nn.sparse_softmax_cross_entropy_with_logits
    xEntropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    loss = tf.reduce_mean(xEntropy, name='loss')

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learningRate)
    trainOp = optimizer.minimize(loss)

# metrics used to evaluate network added here
with tf.name_scope('evaluation'):
    # accuracy
    #@TODO: use softmax instead of logits if other metrics besides
    # correctness are used
    topK = tf.nn.in_top_k(logits, y, 1) # array of batchSize of bools
    accuracy = tf.reduce_mean(tf.cast(topK, tf.float32))

# -------------------- Execution --------------------- #

init = tf.global_variables_initializer()
# saver = tf.train.Saver()

major, minor = utils.read_files('mels/folk')
# testing with major
ngrams = make_ngrams(major, ngramLen)
feedX, feedY = utils.one_hot_ngram(ngrams)
numNGrams = len(ngrams)

Xtrain, Xtest, Ytrain, Ytest = tts(feedX, feedY, test_size=0.2)

nEpochs = 256
batchSize = 128
nBatches = int(np.ceil(len(Xtrain) / batchSize))

with tf.Session() as s:
    init.run()

    for epoch in range(nEpochs):
        Xshuffle, Yshuffle = shuffle(Xtrain, Ytrain) # randomizes batches

        for iteration in range(nBatches):
            Xbatch, Ybatch = next_batch(batchSize, Xshuffle,
                                         Yshuffle, iteration)
            # feed_dict gives values to placeholders
            s.run(trainOp, feed_dict={X: Xbatch, y: Ybatch})

        # evaluates network with entire set of values with current weights
        acc_train = accuracy.eval(feed_dict={X: Xshuffle, y: Yshuffle})
        acc_test = accuracy.eval(feed_dict={X: Xshuffle, y: Yshuffle})

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    # saver.save(s, 'file/path/here')
