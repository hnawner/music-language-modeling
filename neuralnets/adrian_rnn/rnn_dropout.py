#!/usr/bin/env python

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import rnn_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts
from tensorflow.contrib.layers import fully_connected

class RNNDrop:

    n_outputs = 88
    n_inputs = 20

    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.X_train, self.X_test, self.y_train, self.y_test = rnn_utils.setup_rnn(data_folder)
        self.seq_length = rnn_utils.max_length(self.X_train)


    def train_construct(self, architecture, cell_type, p, graph, is_training, 
        learningRate = 0.005):

        with graph.as_default():

            # placeholders
            self.tn_X = tf.placeholder(tf.float32, shape=[None, 
                self.seq_length, RNNDrop.n_inputs], name="X")
            self.tn_y = tf.placeholder(tf.float32, shape=[None, 
                self.seq_length, RNNDrop.n_outputs], name='y')
            self.tn_seq_lens = tf.placeholder(tf.int32, [None])


            cells = []

            for index in range(len(architecture)):
                cell = cell_type(num_units = architecture[index],
                    use_peepholes = True)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = p)
                cells.append(cell)
            
            # build network
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                multi_layer_cell,self.tn_X, dtype = tf.float32, 
                sequence_length = self.tn_seq_lens, swap_memory = True)

            # reshape outputs
            stacked_rnn_outputs = tf.reshape(rnn_outputs, 
                [-1, architecture[(len(architecture) - 1)]])
            stacked_outputs = fully_connected(stacked_rnn_outputs, 
                RNNDrop.n_outputs, activation_fn = None)
            self.tn_outputs = tf.reshape(stacked_outputs, 
                [-1, self.seq_length, RNNDrop.n_outputs])

            loss = self.loss_fn(self.tn_outputs, self.tn_y)
            optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
            self.tn_trainOp = optimizer.minimize(loss)

            
            self.tn_saver = tf.train.Saver()
            self.tn_init = tf.global_variables_initializer()



    def test_construct(self, architecture, cell_type, p, graph, is_training, 
        learningRate = 0.005):

        with graph.as_default():

            # placeholders
            self.ts_X = tf.placeholder(tf.float32, shape=[None, self.seq_length,
                RNNDrop.n_inputs], name="X")
            self.ts_y = tf.placeholder(tf.float32, shape=[None, self.seq_length, 
                RNNDrop.n_outputs], name='y')
            self.ts_seq_lens = tf.placeholder(tf.int32, [None])


            cells = []

            for index in range(len(architecture)):
                cell = cell_type(num_units = architecture[index], 
                    use_peepholes = True)
                cells.append(cell)
            
            # build network
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                multi_layer_cell, self.ts_X, dtype = tf.float32, 
                sequence_length = self.ts_seq_lens, swap_memory = True)

            # reshape outputs
            stacked_rnn_outputs = tf.reshape(rnn_outputs, 
                [-1, architecture[(len(architecture) - 1)]])
            stacked_outputs = fully_connected(stacked_rnn_outputs, 
                RNNDrop.n_outputs, activation_fn = None)
            self.ts_outputs = tf.reshape(stacked_outputs, 
                [-1, self.seq_length, RNNDrop.n_outputs])


            self.ts_log_prob = self.loss_fn(self.ts_outputs, self.ts_y)
            self.ts_accuracy = self.accuracy_fn(self.ts_outputs, self.ts_y)

            self.ts_saver = tf.train.Saver()


    def train_and_test(self, architecture, cell_type, p, nEpochs, batch_size, 
        save_file, learningRate = 0.001):

        self.arch = architecture
        self.cell_type = cell_type

        self.print_details()


        train_graph = tf.Graph()
        test_graph = tf.Graph()

        self.train_construct(architecture, cell_type, p, train_graph, 
            is_training = True)
        self.test_construct(architecture, cell_type, p, test_graph, 
            is_training = False)


        train = tf.Session(graph = train_graph)

        with train as init_sess:
            init_sess.run(self.tn_init)

            nBatches = int(np.ceil(len(self.X_train) / batch_size))
            for iteration in range(nBatches):
                Xbatch, Ybatch = self.next_batch(batch_size, self.X_train, 
                    self.y_train, iteration)
                lengths = rnn_utils.get_mel_lengths(Xbatch)

                init_sess.run(self.tn_trainOp, feed_dict={self.tn_X: Xbatch,
                    self.tn_y: Ybatch, self.tn_seq_lens: lengths})

            self.tn_saver.save(init_sess, save_file)

        for epoch in range(1, nEpochs):

            train = tf.Session(graph = train_graph)

            with train as train_sess:

                self.tn_saver.restore(train_sess, save_file)

                Xshuffle, Yshuffle = shuffle(self.X_train, self.y_train) 

                nBatches = int(np.ceil(len(self.X_train) / batch_size))
                for iteration in range(nBatches):
                    Xbatch, Ybatch = self.next_batch(batch_size, Xshuffle, 
                        Yshuffle, iteration)
                    lengths = rnn_utils.get_mel_lengths(Xbatch)

                    train_sess.run(self.tn_trainOp, feed_dict={
                        self.tn_X: Xbatch, self.tn_y: Ybatch, 
                        self.tn_seq_lens: lengths})

                self.tn_saver.save(train_sess, save_file)


            test = tf.Session(graph = test_graph)

            with test as test_sess:
                
                self.ts_saver.restore(test_sess, save_file)

                train_dict = {self.ts_X: self.X_train, self.ts_y: self.y_train,
                    self.ts_seq_lens:rnn_utils.get_mel_lengths(Xshuffle)}
                test_dict = {self.ts_X : self.X_test, self.ts_y : self.y_test,
                    self.ts_seq_lens: rnn_utils.get_mel_lengths(self.X_test)}

                lp_tn = self.ts_log_prob.eval(feed_dict = train_dict)
                lp_ts = self.ts_log_prob.eval(feed_dict = test_dict)

                acc_tn = self.ts_accuracy.eval(feed_dict = train_dict)
                acc_ts = self.ts_accuracy.eval(feed_dict = test_dict)

                print(epoch, "Train log prob:", lp_tn, "Test log prob:", lp_ts)
                print(epoch, "Train acc:", acc_tn, "Test acc:", acc_ts)



    def print_details(self):
        print("Architecture:", self.arch)
        print("\nCell type:", self.cell_type)
        print("\nMels: ", self.data_folder)


    def loss_fn(self, out, y):
        cross_ent = y * tf.log(tf.nn.softmax(out))
        cross_ent = -tf.reduce_sum(cross_ent, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
        cross_ent = cross_ent * mask 
        cross_ent = tf.reduce_sum(cross_ent, reduction_indices=1)
        cross_ent = cross_ent / tf.reduce_sum(mask, reduction_indices=1)
        return tf.reduce_mean(cross_ent)


    def next_batch(self, size, X, y, iteration):
        start = size * iteration
        stop = start + size
        return X[start:stop], y[start:stop]


    def accuracy_fn(self, out, y):
        out_vals, out_ind = tf.nn.top_k(out, 1)
        y_vals, y_ind = tf.nn.top_k(y, 1)
        compare = tf.equal(out_ind, y_ind)
        mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
        compare = tf.reduce_mean(tf.cast(compare, tf.float32), axis = 2)
        acc = compare * mask 
        acc = tf.reduce_sum(acc, reduction_indices = 1)
        acc = acc / tf.reduce_sum(mask, reduction_indices=1)
        return tf.reduce_mean(acc)


