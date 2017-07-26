#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from crnn_utils import get_data, get_lengths
from sklearn.utils import shuffle
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

TIME_STEP = 125
#KERN_SIZE = 4

# padded with [-1, -1]
path = '/home/hawner2/reu/musical-forms/mels/krn_split/converted/train/'
(X_tr, X_te, Y_tr, Y_te), max_len = get_data(path, TIME_STEP)

n_inputs = 2
n_outputs = 3

# log file
now = datetime.utcnow().strftime("%H%M%S")
root_logdir = 'crnn_rhythm_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

# X lengths after CNN processing is len - kern_size
X = tf.placeholder(tf.float32, [None, max_len, n_inputs], name="X")
Y = tf.placeholder(tf.float32, [None, max_len-7, n_outputs], name="Y")
#lens = tf.placeholder(tf.int32, [None])

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


def f1_build(predictions, targets):
    false_pos = false_neg = true_pos = 0
    for preds, labels in zip(predictions, targets):
        len_no_pad = next((x for x in labels if np.nonzero(x) == 0), len(labels))
        labels = labels[:len_no_pad]
        preds = preds[:len_no_pad]
        _, preds = tf.nn.top_k(preds, 1)
        _, labels = tf.nn.top_k(labels, 1)
        p = np.squeeze(preds.eval(), axis=1)
        l = np.squeeze(labels.eval(), axis=1)
        note_p, note_l = [], []
        for p, l in zip(p, l):
            if l > 0 and note_l != []:
                # find differences
                if note_p == note_l: true_pos += 1
                else:
                    # false pos when predicted note/rest is not in target
                    false_pos += np.count_nonzero(note_p)
                    # false neg when actual note/rest is not predicted
                    false_neg += np.count_nonzero(note_l)
                # reset note
                note_p, note_l  = [p], [l]
            elif l > 0: note_p, note_l = [p], [l]
            else:
                note_p.append(p)
                note_l.append(l)
        # last note
        if note_p == note_l: true_pos += 1
        else:
            false_pos += np.count_nonzero(note_p)
            false_neg += np.count_nonzero(note_l)
    return false_pos, false_neg, true_pos

def f1_score(false_pos, false_neg, true_pos):
    if false_pos == false_neg == true_pos == 0: return 0
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return (2 * precision * recall) / (precision + recall)


learn_rate = 0.05

# loss
loss = loss_fn(logits, Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
save_name = 'crnn-rhythm-train-70e'.format(logdir)
loss_summary = tf.summary.scalar('Loss', loss)

def next_batch(size, x, y, n):
    start = size * n
    end = start + size
    if end >= len(x):
        return x[start:], y[start:] 
    return x[start:end], y[start:end]

n_epochs = 70
batch_size = 32
n_batches = int(np.ceil(len(X_tr) / batch_size))
n_batches_te = int(np.ceil(len(X_te) / batch_size))

with tf.Session() as s:
    init.run()
    for e in range(n_epochs):
        if e % 10 == 0:
            save_path = saver.save(s, save_name)
        X_shuf, Y_shuf = shuffle(X_tr, Y_tr)
        # train
        for b in range(n_batches):
            X_batch, Y_batch = next_batch(batch_size, X_shuf, Y_shuf, b)
            s.run(train_op, feed_dict={X: X_batch, Y: Y_batch})
        # eval
        if e % 10 == 0:
            log_tr, log_te = [], []
            fp_tr = fn_tr = tp_tr = 0
            fp_te = fn_te = tp_te = 0
            for b in range(n_batches):
                X_tr_b, Y_tr_b = next_batch(batch_size, X_tr, Y_tr, b)
                log_tr += [ loss.eval(feed_dict={X: X_tr_b, Y: Y_tr_b}) ]
                logits_tr = logits.eval(feed_dict={X: X_tr_b, Y: Y_tr_b})
                fp, fn, tp = f1_build(logits_tr, Y_tr_b)
                fp_tr += fp
                fn_tr += fn
                tp_tr += tp
            for b in range(n_batches_te):
                X_te_b, Y_te_b = next_batch(batch_size, X_te, Y_te, b)
                log_te += [ loss.eval(feed_dict={X: X_te_b, Y: Y_te_b}) ]
                logits_te = logits.eval(feed_dict={X: X_te_b, Y: Y_te_b})
                fp, fn, tp = f1_build(logits_te, Y_te_b)
                fp_te += fp
                fn_te += fn
                tp_te += tp

            f1_tr = f1_score(fp_tr, fn_tr, tp_tr)
            f1_te = f1_score(fp_te, fn_te, tp_te)

        

        print('------------ %d ------------' % e)
        print('logp -tr', np.mean(log_tr), '-te', np.mean(log_te))
        print('f1   -tr', f1_tr, '-te', f1_te)

    save_path = saver.save(s, save_name)


