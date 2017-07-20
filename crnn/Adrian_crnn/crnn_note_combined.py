#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import nrnn_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from copy import deepcopy
import sys
import numpy as np
import tensorflow as tf

## PITCH AND RHTYHM INTERACT



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


def convert_preds(truth_r, truth_p, pred_r, pred_p):
    truth = []
    pred = []

    scale = 3
    subdiv = 125
    
    for t_r, t_p, p_r, p_p in zip(truth_r, truth_p, pred_r, pred_p):
        p_r = p_r.tolist()
        p_p = p_p.tolist()
        t = []
        p = []
        for step in range(len(t_r)):
            t_val = ([k for k, v in nrnn_utils.r_dict.items() if v == (np.argmax(t_r[step]))])[0]
            t_len = int(round((t_val * scale) / subdiv))
            t_pitch = np.argmax(t_p[step])
            new_t_ts = [t_pitch] + ([0] * t_len)
            t += new_t_ts

            p_val = ([k for k, v in nrnn_utils.r_dict.items() if v == (np.argmax(p_r[step]))])[0]
            p_len = int(round((p_val * scale) / subdiv))
            p_pitch = np.argmax(p_p[step])
            new_p_ts = [p_pitch] + ([0] * p_len)
            p += new_p_ts        
            

        truth.append(t)
        pred.append(p)	
               

    return truth, pred


def compute_f1(preds, labels):

    false_pos = false_neg = true_pos = 0
    for p, l in zip(preds, labels):
        if len(p) > len(l): # needs padding
            diff = len(p) - len(l)
            l += [0] * diff
        elif len(l) > len(p): # needs padding
            diff = len(l) - len(p)
            p += [0] * diff
        note_p, note_l = [], []
        for p, l in zip(p, l):
            if l > 0 and note_l != []:
                # find differences
                if note_p == note_l and p > 0:
                    true_pos += 1
                else:
                    # false pos when predicted note/rest is not in target
                    false_pos += sum(x > 0 for x in note_p)
                    # false neg when actual note/rest is not predicted
                    false_neg += sum(x > 0 for x in note_l)
                # reset note
                note_p = [p]
                note_l = [l]
            elif l > 0:
                note_p = [p]
                note_l = [l]
            else:
                note_p.append(p)
                note_l.append(l)
        # evaluate last note(s)
        if note_p == note_l:    
            true_pos += 1
        else:
            # false pos when predicted note/rest is not in target
            false_pos += sum(x > 0 for x in note_p)
            # false neg when actual note/rest is not predicted
            false_neg += sum(x > 0 for x in note_l)

    if false_pos == 0 or true_pos == 0 or false_neg == 0:
        return 0
    prec =  float(true_pos) / (true_pos + false_pos)
    recall = float(true_pos) / (true_pos + false_neg)
    f1 = (2 * (prec * recall) / (prec + recall))
    return f1
	




print("Get data")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

# GET DATA
path = '/home/aeldrid2/REU/krn_split/converted/train/'
start = 8

# train/test split
X_tr_r, X_te_r, y_tr_r, y_te_r, X_tr_p, X_te_p, y_tr_p, y_te_p = nrnn_utils.setup_rnn(path, 8)


print("Build networks")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
# ##############Network##############

n_p_inputs = 21
n_p_outputs = 95
n_r_inputs = 41
n_r_outputs = 40

X_p = tf.placeholder(tf.float32, [None, None, n_p_inputs], name="X_p")
y_p = tf.placeholder(tf.float32, [None, None, n_p_outputs], name="y_p")

X_r = tf.placeholder(tf.float32, [None, None, n_r_inputs], name="X_r")
y_r = tf.placeholder(tf.float32, [None, None, n_r_outputs], name="Y_r")



# CNN pitch
network_p = tf.layers.conv1d(inputs=X_p,
                           filters=16, # reset to 16
                           kernel_size=4, # reduces size of input
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv_p_1')


network_p = tf.layers.conv1d(inputs=network_p,
                           filters=12, #reset to 16
                           kernel_size=5, # keeps input/output size constant
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv_p_2')


# CNN rhythm
network_r = tf.layers.conv1d(inputs=X_r,
                           filters=16, # reset to 16
                           kernel_size=4, # reduces size of input
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv_r_1')


network_r = tf.layers.conv1d(inputs=network_r,
                           filters=12, #reset to 16
                           kernel_size=5, # keeps input/output size constant
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv_r_2')


# Concat pitch and rhythm
combined = tf.concat([network_p, network_r], axis = 2)


nu_rnn = 64 # rest to 64

# RNN
cells = []
cells.append(tf.contrib.rnn.LSTMCell(nu_rnn, activation=tf.tanh))
cells.append(tf.contrib.rnn.LSTMCell(nu_rnn, activation=tf.tanh))
multi = tf.contrib.rnn.MultiRNNCell(cells)
outs, _ = tf.nn.dynamic_rnn(multi, combined, dtype=tf.float32, swap_memory=True, scope = "rhythm")


# separate pitch
stacked_outs_p = tf.reshape(outs, [-1, nu_rnn], name='stacked_outs_p')
stacked_logits_p = tf.layers.dense(stacked_outs_p, n_p_outputs, name='dense_p')
logits_p = tf.reshape(stacked_logits_p, [-1, tf.shape(y_p)[1], n_p_outputs],
                    name='logits_p')


# separate rhythm
stacked_outs_r = tf.reshape(outs, [-1, nu_rnn], name='stacked_outs_r')
stacked_logits_r = tf.layers.dense(stacked_outs_r, n_r_outputs, name='dense_r')
logits_r = tf.reshape(stacked_logits_r, [-1, tf.shape(y_r)[1], n_r_outputs],
                    name='logits_r')


learn_rate = 0.1

# loss
loss_r = loss_fn(logits_r, y_r)
loss_p = loss_fn(logits_p, y_p)
total_loss = tf.add(loss_r, loss_p)
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
train_op = optimizer.minimize(total_loss)
accuracy_r = accuracy_fn(logits_r, y_r)
accuracy_p = accuracy_fn(logits_p, y_p)




init = tf.global_variables_initializer()



print("Execute")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

def next_batch(size, x, y, n):
    start = size * n
    end = start + size
    if end >= len(x):
        return x[start:], y[start:] 
    return x[start:end], y[start:end]

n_epochs = 129
batch_size = 64
n_batches = int(np.ceil(len(X_tr_r) / batch_size))
n_batches_te = int(np.ceil(len(X_te_r) / batch_size))


with tf.Session() as s:
    init.run()
    for e in range(n_epochs):
        print("Train epoch ", e)
        print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        X_shuf_p, y_shuf_p, X_shuf_r, y_shuf_r = shuffle(X_tr_p, y_tr_p, X_tr_r, y_tr_r)
        for b in range(n_batches):

            # rhythm
            X_batch_r, y_batch_r = next_batch(batch_size, X_shuf_r, y_shuf_r, b)
            rhy_len = nrnn_utils.pad(X_batch_r, y_batch_r, 41, 40)

            # pitch
            X_batch_p, y_batch_p = next_batch(batch_size, X_shuf_p, y_shuf_p, b)
            p_len = nrnn_utils.pad(X_batch_p, y_batch_p, 21, 95)

            s.run(train_op, feed_dict={X_p: X_batch_p, y_p: y_batch_p,
					X_r: X_batch_r, y_r: y_batch_r})


        # Evaluation
        if e == 128:
            print("Evaluate epoch ", e)
            print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            acc_tr_r, acc_te_r, log_tr_r, log_te_r = [], [], [], []
            acc_tr_p, acc_te_p, log_tr_p, log_te_p = [], [], [], []
            total_log_tr, total_log_te = [], []


            truth_tr, pred_tr = [], []
            truth_te, pred_te = [], []

            # train
            for b in range(n_batches):
                X_tr_b_r, y_tr_b_r = next_batch(batch_size, X_tr_r, y_tr_r, b)
                rhy_len = nrnn_utils.pad(X_tr_b_r, y_tr_b_r, 41, 40)
                X_tr_b_p, y_tr_b_p = next_batch(batch_size, X_tr_p, y_tr_p, b)
                p_len = nrnn_utils.pad(X_tr_b_p, y_tr_b_p, 21, 95)
                
                acc_p, log_p, out_p, acc_r, log_r, out_r, t_loss = s.run([accuracy_p, loss_p, logits_p, accuracy_r, loss_r, logits_r, total_loss], feed_dict={X_r: X_tr_b_r, y_r: y_tr_b_r, X_p: X_tr_b_p, y_p: y_tr_b_p})

                acc_tr_r.append(acc_r)
                log_tr_r.append(log_r)
                acc_tr_p.append(acc_p)
                log_tr_p.append(log_p)
                total_log_tr.append(t_loss)

			    # construct 
                truth, pred = convert_preds(y_tr_b_r, y_tr_b_p, out_r, out_p)
                truth_tr += truth
                pred_tr += pred
      
            # test
            for b in range(n_batches_te):
                X_te_b_r, y_te_b_r = next_batch(batch_size, X_te_r, y_te_r, b)
                rhy_len = nrnn_utils.pad(X_te_b_r, y_te_b_r, 41, 40)
                X_te_b_p, y_te_b_p = next_batch(batch_size, X_te_p, y_te_p, b)
                p_len = nrnn_utils.pad(X_te_b_p, y_te_b_p, 21, 95)
                
                acc_r, log_r, out_r, acc_p, log_p, out_p, loss = s.run([accuracy_r, loss_r, logits_r,accuracy_p, loss_p, logits_p, total_loss], feed_dict={X_r: X_te_b_r, y_r: y_te_b_r, X_p: X_te_b_p, y_p: y_te_b_p})
                acc_te_r.append(acc_r)
                log_te_r.append(log_r)
                acc_te_p.append(acc_p)
                log_te_p.append(log_p)
                total_log_te.append(loss)

		

			    # construct 
                truth, pred = convert_preds(y_te_b_r, y_te_b_p, out_r, out_p)
                truth_te += truth
                pred_te += pred


            #print(truth_tr[1])
            #print(pred_tr[1])

            # rhythm f1
            #f1_tr_rhy = f1(logits_tr_rhy, Y_tr_rhy)
            #f1_te_rhy = f1(logits_te_rhy, Y_te_rhy)


		    # total f1
            total_f1_tr = compute_f1(truth_tr, pred_tr)
            total_f1_te = compute_f1(truth_te, pred_te)



            print('------------ %d ------------' % e)
            print('...rhythm...')
            print('logp -train', np.mean(log_tr_r), '-test', np.mean(log_te_r))
            print('acc  -train', np.mean(acc_tr_r), '-test', np.mean(acc_te_r))
            #print('f1  -train', f1_tr_rhy, '-test', f1_te_rhy)

            print('...pitch...')
            print('logp -train', np.mean(log_tr_p), '-test', np.mean(log_te_p))
            print('acc  -train', np.mean(acc_tr_p), '-test', np.mean(acc_te_p))

            print('...combined...')
            #print('total acc  -train', total_acc_tr, '-test', total_acc_te)
            print('total f1  -train', total_f1_tr, '-test', total_f1_te)

print("Done")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
