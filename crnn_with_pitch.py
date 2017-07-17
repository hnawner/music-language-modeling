#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from crnn_utils_with_pitch import get_data
from sklearn.utils import shuffle
import sys
import numpy as np
import tensorflow as tf

TIME_STEP = 125
PITCH_MIN = 40
PITCH_MAX = 90
PITCH_RANGE = PITCH_MAX - PITCH_MIN
#KERN_SIZE = 4

path = '/home/hawner2/reu/musical-forms/mels/krn_split/converted/train/'
(X_tr, X_te, Y_tr, Y_te), max_len = get_data(path, TIME_STEP)

n_inputs = 2 + PITCH_RANGE
n_outputs = n_inputs


# X lengths after CNN processing is len - kern_size
X = tf.placeholder(tf.float32, [None, max_len, n_inputs], name="X")
Y = tf.placeholder(tf.float32, [None, max_len-7, n_outputs], name="Y")

# CNN pre-processing
network = tf.layers.conv1d(inputs=X,
                           filters=8,
                           kernel_size=4,
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv1')

network = tf.layers.conv1d(inputs=network,
                           filters=8,
                           kernel_size=5,
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv2')

nu_rnn = 32

# RNN
cells = []
cells.append(tf.contrib.rnn.LSTMCell(nu_rnn, activation=tf.tanh))
cells.append(tf.contrib.rnn.LSTMCell(nu_rnn, activation=tf.tanh))
multi = tf.contrib.rnn.MultiRNNCell(cells)
outs, _ = tf.nn.dynamic_rnn(multi, network, dtype=tf.float32, swap_memory=True)
stacked_outs = tf.reshape(outs, [-1, nu_rnn], name='stacked_outs')
stacked_logits = tf.layers.dense(stacked_outs, n_outputs, name='dense')
logits = tf.reshape(stacked_logits, [-1, max_len-7, n_outputs],
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

'''
def f1(logits, Y):
    f1 = list()
    for p, y in zip(logits, Y):
        tp = fp = fn = 0
        for p_ts, y_ts in zip(p, y):
            if y == [0,0,0]: # padding
                break
            p_ind = np.argmax(p_ts)
            y_ind = np.argmax(y_ts)
            if p_ind == 0 and y_ind == 0:
                tp += 1
            elif p_ind == 0 and y_ind != 0:
                fp += 1
            elif p_ind != 0 and y_ind == 0:
                fp += 1

        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        f_score = 2 * ((precision * recall) / (precision + recall))
        f1.append(f_score)
    return np.mean(f1)
'''

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

n_epochs = 1
batch_size = 32
n_batches = int(np.ceil(len(X_tr) / batch_size))
n_batches_te = int(np.ceil(len(X_te) / batch_size))

with tf.Session() as s:
    init.run()
    # test_lens = get_lengths(Y_te)
    for e in range(n_epochs):
        X_shuf, Y_shuf = shuffle(X_tr, Y_tr)
        # train
        for b in range(n_batches):
            X_batch, Y_batch = next_batch(batch_size, X_shuf, Y_shuf, b)
            batch_lens = get_lengths(Y_batch)
            # train_lens = get_lengths(Y_tr)
            s.run(train_op, feed_dict={X: X_batch, Y: Y_batch})
        # eval
        acc_tr, acc_te, log_tr, log_te = [], [], [], []
        logits_tr, logits_te = [], []
        for b in range(n_batches):
            X_tr_b, Y_tr_b = next_batch(batch_size, X_tr, Y_tr, b)
            acc_tr += [ accuracy.eval(feed_dict={X: X_tr_b, Y: Y_tr_b}) ]
            log_tr += [ loss.eval(feed_dict={X: X_tr_b, Y: Y_tr_b}) ]
        for b in range(n_batches_te):
            X_te_b, Y_te_b = next_batch(batch_size, X_te, Y_te, b)
            acc_te += [ accuracy.eval(feed_dict{X: X_te_b, Y: Y_te_b}) ]
            log_te += [ loss.eval(feed_dict={X: X_te_b, Y: Y_te_b}) ]
            # F1 score
            # logits_tr += [ logits.eval(feed_dict={X: X_tr_b, Y: Y_tr_b}) ]
            # logits_te += [ logits.eval(feed_dict={X: X_te_b, Y: Y_te_b}) ]
        # f1_tr = f1(logits_tr, Y_tr)
        # f1_te = f1(logits_te, Y_te)
        

        print('------------ %d ------------' % e)
        print('logp -tr', np.mean(log_tr), '-te', np.mean(log_te))
        print('acc  -tr', np.mean(acc_tr), '-te', np.mean(acc_te))
        # print('f1   -tr', f1_tr, '-te', f1_te)


