#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from crnn_utils import get_data, get_lengths
from sklearn.utils import shuffle
import sys
import numpy as np
import tensorflow as tf

TIME_STEP = sys.argv[1]
KERN_SIZE = sys.argv[2]

# padded with [-1, -1]
path = '/home/hawner2/reu/musical-forms/mels/krn_split/converted/train/'
(X_tr, X_te, Y_tr, Y_te), max_len = get_data(path, TIME_STEP, KERN_SIZE)

kern_size = KERN_SIZE
n_inputs = 2
n_outputs = 3

# X lengths after CNN processing is len - kern_size
X = tf.placeholder(tf.float32, [None, max_len, n_inputs], name="X")
Y = tf.placeholder(tf.float32, [None, max_len-kern_size+1, n_outputs], name="Y")
lens = tf.placeholder(tf.int32, [None])

# CNN pre-processing
network = tf.layers.conv1d(inputs=X,
                           filters=16,
                           kernel_size=kern_size, # reduces size of input
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv1')

network = tf.layers.conv1d(inputs=network,
                           filters=8,
                           kernel_size=1, # keeps input/output size constant
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv2')

nu_rnn = 64

# RNN
cells = []
cells.append(tf.contrib.rnn.LSTMCell(nu_rnn, activation=tf.tanh))
cells.append(tf.contrib.rnn.LSTMCell(nu_rnn, activation=tf.tanh))
multi = tf.contrib.rnn.MultiRNNCell(cells)
outs, _ = tf.nn.dynamic_rnn(multi, network, sequence_length=lens,
                            dtype=tf.float32, swap_memory=True)
stacked_outs = tf.reshape(outs, [-1, nu_rnn], name='stacked_outs')
stacked_logits = tf.layers.dense(stacked_outs, n_outputs, name='dense')
logits = tf.reshape(stacked_logits, [-1, max_len-kern_size+1, n_outputs],
                    name='logits')

def loss_fn(logits, Y):
    xentropy = Y * tf.log(tf.nn.softmax(logits))
    xentropy = -tf.reduce_sum(xentropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(Y), reduction_indices=2))
    xentropy *= mask
    xentropy = tf.reduce_sum(xentropy, reduction_indices=1)
    xentropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(xentropy)

def accuracy_fn(logits, Y):
    _, l_inds = tf.nn.top_k(logits, 1)
    _, y_inds = tf.nn.top_k(Y, 1)
    comparison = tf.equal(l_inds, y_inds)
    comparison = tf.reduce_mean(tf.cast(comparison, tf.float32), axis=2)
    mask = tf.sign(tf.reduce_max(tf.abs(Y), reduction_indices=2))
    acc = comparison * mask
    acc = tf.reduce_sum(acc, reduction_indices=1)
    acc /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(acc)

learn_rate = 0.1

# loss
loss = loss_fn(logits, Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
train_op = optimizer.minimize(loss)
accuracy = accuracy_fn(logits, Y)

init = tf.global_variables_initializer()

def next_batch(size, x, y, n):
    start = size * n
    end = start + size
    if end >= len(x):
        return x[start:], y[start:] 
    return x[start:end], y[start:end]

n_epochs = 100
batch_size = 32
n_batches = int(np.ceil(len(X_tr) / batch_size))

with tf.Session() as s:
    init.run()
    test_lens = get_lengths(Y_te)
    for e in range(n_epochs):
        X_shuf, Y_shuf = shuffle(X_tr, Y_tr)
        for b in range(n_batches):
            X_batch, Y_batch = next_batch(batch_size, X_shuf, Y_shuf, b)
            batch_lens = get_lengths(Y_batch)
            train_lens = get_lengths(Y_tr)
            s.run(train_op, feed_dict={X: X_batch, Y: Y_batch, lens: batch_lens})
        acc_tr = accuracy.eval(feed_dict={X: X_tr, Y: Y_tr, lens: train_lens})
        acc_te = accuracy.eval(feed_dict={X: X_te, Y: Y_te, lens: test_lens})
        log_tr = loss.eval(feed_dict={X: X_tr, Y: Y_tr, lens: train_lens})
        log_te = loss.eval(feed_dict={X: X_te, Y: Y_te, lens: test_lens})

        print('------------ %d ------------' % e)
        print('logp -tr', log_tr, '-te', log_te)
        print('acc  -tr', acc_tr, '-te', acc_te)


