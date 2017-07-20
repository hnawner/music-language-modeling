#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import crnn_utils02 as crnn_utils
import rnn_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from copy import deepcopy
import sys
import numpy as np
import tensorflow as tf


# Params:
TIME_STEP = 250

'''
def loss_fn(logits, Y):
    xentropy = Y * tf.log(tf.nn.softmax(logits))
    xentropy = -tf.reduce_sum(xentropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(Y), reduction_indices=2))
    xentropy *= mask
    xentropy = tf.reduce_sum(xentropy, reduction_indices=1)
    xentropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(xentropy)
'''



def loss_fn(logits, Y, net, weights = None):
    xentropy = Y * tf.log(tf.nn.softmax(logits))
    if net == "rhythm":
        xentropy *= weights
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


def construct_preds(truth_rhy, truth_p, pred_rhy, pred_p):
    truth = []
    pred = []

    
    for t_rhy, t_p, p_rhy, p_p in zip(truth_rhy, truth_p, pred_rhy, pred_p):
        p_rhy = p_rhy.tolist()
        p_p = p_p.tolist()
        t = []
        p = []
        for step in range(len(t_rhy)):
            if sum(t_rhy[step]) == 0: # padding
                break
            elif np.argmax(t_rhy[step]) == 0: # start of note
                if t_p != []:
                    t.append(np.argmax(t_p.pop(0)))
            elif np.argmax(t_rhy[step]) == 1: # continue note
                    t.append(0)
            else:
                t.append(np.argmax(t_rhy[step]))
           
            if np.argmax(p_rhy[step]) == 0:
                if p_p != []:
                    p.append(np.argmax(p_p.pop(0)))
            elif np.argmax(p_rhy[step]) == 1: # continue note
                    p.append(0)
            else:
                p.append(np.argmax(p_rhy[step]))

        truth.append(t)
        pred.append(p)	
               

    return truth, pred


def compute_f1(preds, labels):

    false_pos = false_neg = true_pos = 0
    for p, l in zip(preds, labels):
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
rhythms, pitches, files = crnn_utils.read(path)


# rhythm:
X_rhy = rhythms
X_rhy = crnn_utils.encode_input(deepcopy(X_rhy), files, 250, 16)
for i in range(len(X_rhy)):
    X_rhy[i] = X_rhy[i][:-1]
Y_rhy = crnn_utils.encode_target(rhythms, files, 250)




# pitches:
X_p = rnn_utils.rnn_encoder(pitches)
for i in range(len(X_p)):
    X_p[i] = X_p[i][:-1]
Y_p = rnn_utils.rnn_labels_encoder(pitches)

# train/test split
X_tr_rhy, X_te_rhy, Y_tr_rhy, Y_te_rhy, X_tr_p, X_te_p, Y_tr_p, Y_te_p = train_test_split(X_rhy, Y_rhy, X_p, Y_p, test_size=0.2)


print("Build networks")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
# ##############RHYTHM##############

#kern_size = KERN_SIZE
n_inputs = 3
n_outputs = 2

X = tf.placeholder(tf.float32, [None, None, n_inputs], name="X_rhy")
Y = tf.placeholder(tf.float32, [None, None, n_outputs], name="Y_rhy")
loss_weights = tf.placeholder(tf.float32, [None, None, n_outputs])



# CNN pre-processing
network = tf.layers.conv1d(inputs=X,
                           filters=16, # reset to 16
                           kernel_size=8, # reduces size of input
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv1')




network = tf.layers.conv1d(inputs=network,
                           filters=12, #reset to 16
                           kernel_size=9, # keeps input/output size constant
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv2')


nu_rnn = 64 # rest to 64

# RNN
cells = []
cells.append(tf.contrib.rnn.LSTMCell(nu_rnn, activation=tf.tanh))
cells.append(tf.contrib.rnn.LSTMCell(nu_rnn, activation=tf.tanh))
multi = tf.contrib.rnn.MultiRNNCell(cells)
outs, _ = tf.nn.dynamic_rnn(multi, network, dtype=tf.float32, swap_memory=True, scope = "rhythm")
stacked_outs = tf.reshape(outs, [-1, nu_rnn], name='stacked_outs')
stacked_logits = tf.layers.dense(stacked_outs, n_outputs, name='dense')
logits = tf.reshape(stacked_logits, [-1, tf.shape(Y)[1], n_outputs],
                    name='logits')

learn_rate = 0.1

# loss
loss_rhy = loss_fn(logits, Y, "rhythm", loss_weights)
optimizers_rhy = tf.train.AdamOptimizer(learning_rate=learn_rate)
train_ops_rhy = optimizers_rhy.minimize(loss_rhy)
accuracy_rhy = accuracy_fn(logits, Y)






# ##############Pitch##############

n_inputs_p = 21
n_outputs_p = 95

#max_len_p = tf.placeholder(tf.int32)
#X_p = tf.placeholder(tf.float32, shape=[None, max_len_p, n_inputs_p], name="X_p")
#Y_p = tf.placeholder(tf.float32, shape=[None, max_len_p, n_outputs_p], name='y_p')

X_p = tf.placeholder(tf.float32, shape=[None, None, n_inputs_p], name="X_p")
Y_p = tf.placeholder(tf.float32, shape=[None, None, n_outputs_p], name='y_p')


nu_rnn_p = 64

# RNN
cells_p = []
cells_p.append(tf.contrib.rnn.LSTMCell(nu_rnn_p, activation=tf.tanh))
cells_p.append(tf.contrib.rnn.LSTMCell(nu_rnn_p, activation=tf.tanh))
        
multi_layer_cell_p = tf.contrib.rnn.MultiRNNCell(cells_p)
rnn_outputs_p, rnn_states_p = tf.nn.dynamic_rnn(multi_layer_cell_p, X_p, 
                dtype = tf.float32, swap_memory = True, scope = "pitch")

stacked_rnn_outputs_p = tf.reshape(rnn_outputs_p, [-1, nu_rnn_p])
stacked_outputs_p = tf.contrib.layers.fully_connected(stacked_rnn_outputs_p, n_outputs_p, activation_fn = None)
outputs_p = tf.reshape(stacked_outputs_p, [-1, tf.shape(Y_p)[1], n_outputs_p])

# Loss
loss_p = loss_fn(outputs_p, Y_p, "pitch")
optimizer_p = tf.train.AdamOptimizer(learning_rate = 0.1)
train_ops_p = optimizer_p.minimize(loss_p)
accuracy_p = accuracy_fn(outputs_p, Y_p)





init = tf.global_variables_initializer()










print("Execute")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

def next_batch(size, x, y, n):
    start = size * n
    end = start + size
    if end >= len(x):
        return x[start:], y[start:] 
    return x[start:end], y[start:end]

n_epochs = 65
batch_size = 64
scale_factor = 0.5
n_batches = int(np.ceil(len(X_tr_rhy) / batch_size))
n_batches_te = int(np.ceil(len(X_te_rhy) / batch_size))


with tf.Session() as s:
    init.run()
    for e in range(n_epochs):
        print("Train epoch ", e)
        print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        X_shuf_rhy, Y_shuf_rhy, X_shuf_p, Y_shuf_p = shuffle(X_tr_rhy, Y_tr_rhy, X_tr_p, Y_tr_p)
        for b in range(n_batches):
            # rhythm

            X_batch_rhy, Y_batch_rhy = next_batch(batch_size, X_shuf_rhy, Y_shuf_rhy, b)


            rhy_len = crnn_utils.pad(X_batch_rhy, Y_batch_rhy)
            loss_w = [ [[1, scale_factor]] *len(Y_batch_rhy[0]) ] * len(Y_batch_rhy)


            s.run(train_ops_rhy, feed_dict={X: X_batch_rhy, Y: Y_batch_rhy, loss_weights:loss_w})

            # pitch
            X_batch_p, Y_batch_p = next_batch(batch_size, X_shuf_p, Y_shuf_p, b)
            p_len = rnn_utils.pad(X_batch_p, Y_batch_p)

            s.run(train_ops_p, feed_dict={X_p: X_batch_p, Y_p: Y_batch_p})


        # Evaluation
        if e % 16 == 0:
            print("Evaluate epoch ", e)
            print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            acc_tr_rhy, acc_te_rhy, log_tr_rhy, log_te_rhy = [], [], [], []
            acc_tr_p, acc_te_p, log_tr_p, log_te_p = [], [], [], []

            logits_tr_rhy, logits_te_rhy = [], []
            logits_tr_p, logits_te_p = [], []

            truth_tr, pred_tr = [], []
            truth_te, pred_te = [], []

            # train
            for b in range(n_batches):
                # rhythm train
                X_tr_b_rhy, Y_tr_b_rhy = next_batch(batch_size, X_tr_rhy, Y_tr_rhy, b)
                rhy_len = crnn_utils.pad(X_tr_b_rhy, Y_tr_b_rhy)
                loss_w = [ [[1, scale_factor]] * len(Y_tr_b_rhy[0]) ] * len(Y_tr_b_rhy)

                
                acc_tr_rhy, log_tr_rhy, logits_tr_rhy = s.run([accuracy_rhy, loss_rhy, logits], 
                        feed_dict={X: X_tr_b_rhy, Y: Y_tr_b_rhy, loss_weights:loss_w})

			
                # pitch train
                X_tr_b_p, Y_tr_b_p = next_batch(batch_size, X_tr_p, Y_tr_p, b)
                p_len = rnn_utils.pad(X_tr_b_p, Y_tr_b_p)


                acc_tr_p, log_tr_p, logits_tr_p = s.run([accuracy_p, loss_p, outputs_p], 
                        feed_dict={X_p: X_tr_b_p, Y_p: Y_tr_b_p})


			    # construct 
                truth, pred = construct_preds(Y_tr_b_rhy, deepcopy(Y_tr_b_p), logits_tr_rhy, 
                                    logits_tr_p)
                truth_tr += truth
                pred_tr += pred
      
            # test
            for b in range(n_batches_te):
                # rhythm test
                X_te_b_rhy, Y_te_b_rhy = next_batch(batch_size, X_te_rhy, Y_te_rhy, b)
                rhy_len = crnn_utils.pad(X_te_b_rhy, Y_te_b_rhy)
                loss_w = [ [[1, scale_factor]] *len(Y_te_b_rhy[0]) ] * len(Y_te_b_rhy)
                
                acc_te_rhy, log_te_rhy, logits_te_rhy = s.run([accuracy_rhy, loss_rhy, logits], 
                        feed_dict={X: X_te_b_rhy, Y: Y_te_b_rhy, loss_weights:loss_w})
      
                # pitch test
                X_te_b_p, Y_te_b_p = next_batch(batch_size, X_te_p, Y_te_p, b)
                p_len = rnn_utils.pad(X_te_b_p, Y_te_b_p)
                
                acc_te_p, log_te_p, logits_te_p = s.run([accuracy_p, loss_p, outputs_p], 
                        feed_dict={X_p: X_te_b_p, Y_p: Y_te_b_p})

			    # construct 
                truth, pred = construct_preds(Y_te_b_rhy, deepcopy(Y_te_b_p), logits_te_rhy, 
                                    logits_te_p)
                truth_te += truth
                pred_te += pred


            print(truth_tr[1])
            print(pred_tr[1])

            # rhythm f1
            #f1_tr_rhy = f1(logits_tr_rhy, Y_tr_rhy)
            #f1_te_rhy = f1(logits_te_rhy, Y_te_rhy)


		    # total f1
            total_f1_tr = compute_f1(truth_tr, pred_tr)
            total_f1_te = compute_f1(truth_te, pred_te)



            print('------------ %d ------------' % e)
            print('...rhythm...')
            print('logp -train', np.mean(log_tr_rhy), '-test', np.mean(log_te_rhy))
            print('acc  -train', np.mean(acc_tr_rhy), '-test', np.mean(acc_te_rhy))
            #print('f1  -train', f1_tr_rhy, '-test', f1_te_rhy)

            print('...pitch...')
            print('logp -train', np.mean(log_tr_p), '-test', np.mean(log_te_p))
            print('acc  -train', np.mean(acc_tr_p), '-test', np.mean(acc_te_p))

            print('...combined...')
            #print('total acc  -train', total_acc_tr, '-test', total_acc_te)
            print('total f1  -train', total_f1_tr, '-test', total_f1_te)

print("Done")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
