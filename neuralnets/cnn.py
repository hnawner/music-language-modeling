#!/usr/bin/env python

import utils
#from __future__ import division, print_function
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

class CNN:

    pitchRange = 88
    nOutputs = 88

    def __init__(self, ngramLen, path, mode):
        self.ngramLen = ngramLen
        self.mode = mode
        self.train_X, self. test_X, self.train_y, self.test_y = utils.setup_CNN(path, self.ngramLen, self.mode)
        self.trainLen, self.height, self.nWidth, self.nChan = np.shape(self.train_X)
        self.X = tf.placeholder(tf.float32, shape = (None, self.height, self.nWidth, self.nChan), name = "X")
        self.y = tf.placeholder(tf.int64, shape = (None), name = "y")
        

    def fully_conn_layer(self, X, n_neurons, act = None):
        with tf.name_scope("fc"):

            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
            W = tf.Variable(init, name = "weights")
            b = tf.Variable(tf.zeros([n_neurons]), name = "biases")
            z = tf.matmul(X, W) + b
            return act(z) if act != None else z
      
    def construct(self):		
        with tf.name_scope("cnn"):

            # Convolotional layer
            conv1 = tf.layers.conv2d(self.X, 
                    filters = 32, 
                    kernel_size = [1, 12],
                    strides = [1, 1],                     
                    padding = "VALID", 
                    activation = tf.nn.relu)
            print("Conv1 shape:")
            print(conv1.get_shape())

            # Pooling layer
            mPool1 = tf.nn.max_pool(conv1, 
                        ksize = [1,1,4,1], 
                        strides = [1,1,4,1], 
                        padding = "SAME", 
                        name ="mPool1")
            print("Pooling1 shape:")
            print(mPool1.get_shape())

            # Convolotional layer
            conv2 = tf.layers.conv2d(mPool1, 
                    filters = 16, 
                    kernel_size = [1, 6],
                    strides = [1, 1],                     
                    padding = "SAME", 
                    activation = tf.nn.relu)
            print("Conv2 shape:")
            print(conv2.get_shape())

            # Pooling layer
            mPool2 = tf.nn.max_pool(conv2, 
                        ksize = [1,1,4,1], 
                        strides = [1,1,4,1], 
                        padding = "SAME", 
                        name ="mPool1")
            print("Pooling1 shape:")
            print(mPool1.get_shape())

            # Flatten
            dims = mPool2.get_shape().as_list()
            newDim = dims[1] * dims[2] * dims[3]
            mPool2_flat = tf.reshape(tensor = mPool2, shape = [-1, newDim])
            print("Flattened shape:")
            print(mPool2_flat.get_shape())

            # Fully connected layer
            fc1 = self.fully_conn_layer(mPool2_flat,
                         n_neurons = 256, 
                         act = tf.nn.tanh)
            print("FC1 shape:")
            print(fc1.get_shape())

            # Fully Connected layer
            fc2 = self.fully_conn_layer(fc1, 
                    n_neurons = 512, 
                    act = tf.nn.tanh)
            print("FC2 shape:")
            print(fc2.get_shape())

            # Fully Connected layer
            fc3 = self.fully_conn_layer(fc2, 
                    n_neurons = 256, 
                    act = tf.nn.tanh)
            print("FC3 shape:")
            print(fc3.get_shape())


            # Output layer
            logits = self.fully_conn_layer(fc3, CNN.nOutputs)
            print(logits.get_shape())

	
        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y, logits = logits)
            loss = tf.reduce_mean(xentropy, name = "loss")
	
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.training_op = optimizer.minimize(loss)
	
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	
        
    #returns next batch of specified size
    def next_batch(self, size, X, y, iteration):
        start = size * iteration
        stop = start + size
        return X[start:stop], y[start:stop]

    def execute(self, batch_size, nEpochs):
        init = tf.global_variables_initializer()
        n_batches = int(np.ceil(len(self.train_X) / batch_size))
        with tf.Session() as sess:
            init.run()
            for epoch in range(nEpochs):
                X_shuffle, y_shuffle = shuffle(self.train_X, self.train_y)
                for iteration in range(n_batches):
                    X_batch, y_batch = self.next_batch(batch_size, X_shuffle, y_shuffle, iteration)
                    sess.run(self.training_op, feed_dict = {self.X: X_batch, self.y: y_batch})
                acc_train = self.accuracy.eval(feed_dict = {self.X: self.train_X, self.y: self.train_y})
                acc_test = self.accuracy.eval(feed_dict = {self.X: self.test_X, self.y: self.test_y})
                print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
  




