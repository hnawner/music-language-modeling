#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import crnn_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from copy import deepcopy
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import fully_connected

## PITCH AND RHTYHM SEPARATE

## HELPER AND EVALUATION FUNCTIONS ##

# softmax cross entropy
def loss_fn(logits, Y):
    xentropy = Y * tf.log(tf.nn.softmax(logits))
    xentropy = -tf.reduce_sum(xentropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(Y), reduction_indices=2))
    xentropy *= mask
    xentropy = tf.reduce_sum(xentropy, reduction_indices=1)
    xentropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(xentropy)


# accuracy 
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


# converts the outputs of the neural net and the targets 
# from one-hot vectors to note representation
def out2notes(truth_r, truth_p, pred_r, pred_p):
    conv_truth_r = []
    conv_truth_p = []
    conv_pred_r = []
    conv_pred_p = []

    for t_r, t_p, p_r, p_p in zip(truth_r, truth_p, pred_r, pred_p):
        t_r = [np.argmax(i) for i in t_r if sum(i) != 0]
        t_p = [np.argmax(i) for i in t_p if sum(i) != 0]
        p_r = ([np.argmax(i) for i in p_r])[:len(t_r)]
        p_p = ([np.argmax(i) for i in p_p])[:len(t_p)]

        conv_truth_r.append(t_r)
        conv_truth_p.append(t_p)
        conv_pred_r.append(p_r)
        conv_pred_p.append(p_p)

    return conv_truth_r, conv_truth_p, conv_pred_r, conv_pred_p


# computes accuracy predictions in note representation
def total_acc(truth_r, truth_p, pred_r, pred_p):
    total = 0
    correct = 0
    for t_r, t_p, p_r, p_p in zip(truth_r, truth_p, pred_r, pred_p):
        for i in range(len(t_r)):
            if (t_r[i] == p_r[i]) and (t_p[i] == p_p[i]):
                correct += 1
            total += 1

    acc= correct / total

    return acc            



## GET DATA ##

train_path = '/home/aeldrid2/REU/krn_split/converted/other/debug/'
test_path = "/home/aeldrid2/REU/krn_split/converted/other/baby"

start = 12 # length of cnn window in notes
trans = True # whether to transpose all mels to same key

# training data
# X = input, y = target
# tr = train, te = test
# r = rhythm, p = pitch
(X_tr_r, X_te_r, y_tr_r, y_te_r, X_tr_p, X_te_p, y_tr_p, y_te_p), \
		(p_max, p_min) = crnn_utils.setup_rnn(train_path, start, trans)

# data set sizes
train_size = len(X_tr_r)
test_size = len(X_te_r)

# final test data
X_p_final, y_p_final, X_r_final, y_r_final = \
        crnn_utils.setup_test(test_path, p_max, p_min, start, trans)


## BUILD NETWORK ##

# p = pitch, r = rhythm
n_p_inputs = len(X_tr_p[0][0])
n_p_outputs = len(y_tr_p[0][0])
n_r_inputs = len(X_tr_r[0][0])
n_r_outputs = len(y_tr_r[0][0])

X_p = tf.placeholder(tf.float32, [None, None, n_p_inputs], name="X_p")
y_p = tf.placeholder(tf.float32, [None, None, n_p_outputs], name="y_p")

X_r = tf.placeholder(tf.float32, [None, None, n_r_inputs], name="X_r")
y_r = tf.placeholder(tf.float32, [None, None, n_r_outputs], name="Y_r")


# CNN pitch
network_p = tf.layers.conv1d(inputs=X_p,
                           filters=12, 
                           kernel_size=8,
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv_p_1')

network_p = tf.layers.conv1d(inputs=network_p,
                           filters=12, 
                           kernel_size=5, 
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv_p_2')

# CNN rhythm
network_r = tf.layers.conv1d(inputs=X_r,
                           filters=12, 
                           kernel_size=8, 
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv_r_1')

network_r = tf.layers.conv1d(inputs=network_r,
                           filters=12,
                           kernel_size=5,
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv_r_2')


# RNN
nu_rnn = 64

# RNN pitch
cells_p = []
cells_p.append(tf.contrib.rnn.LSTMCell(nu_rnn, use_peepholes = True, 
		activation=tf.tanh))
cells_p.append(tf.contrib.rnn.LSTMCell(nu_rnn, use_peepholes = True, 
		activation=tf.tanh))
multi_p = tf.contrib.rnn.MultiRNNCell(cells_p)
outs_p, _ = tf.nn.dynamic_rnn(multi_p, network_p, dtype=tf.float32,
        swap_memory=True, scope = "pitch")

# RNN rhythm
cells_r = []
cells_r.append(tf.contrib.rnn.LSTMCell(nu_rnn, use_peepholes = True, 
		activation=tf.tanh))
cells_r.append(tf.contrib.rnn.LSTMCell(nu_rnn, use_peepholes = True, 
		activation=tf.tanh))
multi_r = tf.contrib.rnn.MultiRNNCell(cells_r)
outs_r, _ = tf.nn.dynamic_rnn(multi_r, network_r, dtype=tf.float32,
        swap_memory=True, scope = "rhythm")


# batch normalization parameters
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

bn_params = {"is_training": is_training, "decay": 0.99,
        "updates_collections":None, "scale":True}
bn_params_out = {"is_training": is_training,
        "decay": 0.99, "updates_collections":None}

# fully connected pitch
n_hidden_1_p = 48
n_hidden_2_p = 32
keep_prob_p = 0.65

stacked_outs_p = tf.reshape(outs_p, [-1, nu_rnn], name='stacked_outs_p')

stacked_logits_p1 = fully_connected(stacked_outs_p, 
			num_outputs = n_hidden_1_p,
			activation_fn = tf.nn.elu,
			normalizer_fn = batch_norm, 
			normalizer_params=bn_params, 
			scope='dense_p1')

p_drop1 = tf.contrib.layers.dropout(stacked_logits_p1, 
        keep_prob_p, 
        is_training = is_training)

stacked_logits_p2 = fully_connected(p_drop1, 
			num_outputs = n_hidden_2_p, 
			activation_fn = tf.nn.elu,
			normalizer_fn = batch_norm, 
			normalizer_params=bn_params, 
			scope='dense_p2')

p_drop2 = tf.contrib.layers.dropout(stacked_logits_p2, 
        keep_prob_p, 
        is_training = is_training)

stacked_logits_p3 = fully_connected(p_drop2, 
			num_outputs = n_p_outputs,
			activation_fn = None,
			normalizer_fn = batch_norm, 
			normalizer_params=bn_params_out, 
			scope='dense_p_out')

logits_p = tf.reshape(stacked_logits_p3,
        [-1, tf.shape(y_p)[1], n_p_outputs],name='logits_p')

# fully connected rhythm
n_hidden_1_r = 48
n_hidden_2_r = 32
keep_prob_r = 0.65

stacked_outs_r = tf.reshape(outs_r, [-1, nu_rnn], name='stacked_outs_r')

stacked_logits_r1 = fully_connected(stacked_outs_r, 
			n_hidden_1_r, 
			activation_fn = tf.nn.elu,
			normalizer_fn = batch_norm, 
			normalizer_params=bn_params, 
			scope='dense_r1')

r_drop1 = tf.contrib.layers.dropout(stacked_logits_r1, 
        keep_prob_r, 
        is_training = is_training)

stacked_logits_r2 = fully_connected(r_drop1, 
			n_hidden_2_r, 
			activation_fn = tf.nn.elu,
			normalizer_fn = batch_norm, 
			normalizer_params=bn_params, 
			scope='dense_r2')

r_drop2 = tf.contrib.layers.dropout(stacked_logits_r2, 
        keep_prob_r, 
        is_training = is_training)

stacked_logits_r3 = fully_connected(r_drop2, 
			n_r_outputs, 
			activation_fn = None,
			normalizer_fn = batch_norm, 
			normalizer_params=bn_params_out, 
			scope='dense_r_out')

logits_r = tf.reshape(stacked_logits_r3, 
        [-1, tf.shape(y_r)[1], n_r_outputs], name='logits_r')


# loss params
learn_rate = 0.02
clip = 5

# loss
loss_r = loss_fn(logits_r, y_r)
loss_p = loss_fn(logits_p, y_p)
total_loss = tf.add(loss_r, loss_p)

# pitch optimizer with gradient clipping
optimizer_p = tf.train.AdamOptimizer(learning_rate=learn_rate)
gradients_p, variables_p = zip(*optimizer_p.compute_gradients(loss_p))
gradients_p, _ = tf.clip_by_global_norm(gradients_p, clip)
train_op_p = optimizer_p.apply_gradients(zip(gradients_p, variables_p))

# rhythm optimizer with gradient clipping
optimizer_r = tf.train.AdamOptimizer(learning_rate=learn_rate)
gradients_r, variables_r = zip(*optimizer_r.compute_gradients(loss_r))
gradients_r, _ = tf.clip_by_global_norm(gradients_r, clip)
train_op_r = optimizer_r.apply_gradients(zip(gradients_r, variables_r))


# evaluation
accuracy_r = accuracy_fn(logits_r, y_r)
accuracy_p = accuracy_fn(logits_p, y_p)


init = tf.global_variables_initializer()

saver = tf.train.Saver()



## EXECUTION PHASE ##

# returns next batch
def next_batch(size, x, y, n):
    start = size * n
    end = start + size
    if end >= len(x):
        return x[start:], y[start:] 
    return x[start:end], y[start:end]

# execution params
n_epochs = 15
batch_size = 32
eval_batch_size = 32 # larger to improve speed
n_batches = int(np.ceil(len(X_tr_r) / batch_size))
n_batches_tr = int(np.ceil(len(X_tr_r) / eval_batch_size))
n_batches_te = int(np.ceil(len(X_te_r) / eval_batch_size))

# run session
with tf.Session() as s:

    init.run()

    best_loss = sys.maxint

    for e in range(n_epochs):

        # shuffle training set each epoch
        X_shuf_p, y_shuf_p, X_shuf_r, y_shuf_r = shuffle(X_tr_p, y_tr_p, 
                X_tr_r, y_tr_r)

        for b in range(n_batches):

            # rhythm
            X_batch_r, y_batch_r = next_batch(batch_size, 
                    X_shuf_r, y_shuf_r, b)
            rhy_len = crnn_utils.pad(X_batch_r, y_batch_r, 
                    n_r_inputs, n_r_outputs)

            # pitch
            X_batch_p, y_batch_p = next_batch(batch_size,
                    X_shuf_p, y_shuf_p, b)
            p_len = crnn_utils.pad(X_batch_p, y_batch_p, 
                    n_p_inputs, n_p_outputs)

            # training op pitch
            s.run(train_op_p, feed_dict={X_p: X_batch_p, 
					y_p: y_batch_p, is_training: True})

            # training op rhythm
            s.run(train_op_r, feed_dict={X_r: X_batch_r,
					y_r: y_batch_r, is_training: True})

        # evaluate total loss every epoch
        # rhythm
        X_te_b_r, y_te_b_r = next_batch(test_size, X_te_r, y_te_r, 0)
        rhy_len = crnn_utils.pad(X_te_b_r, y_te_b_r, n_r_inputs, n_r_outputs)

        # pitch
        X_te_b_p, y_te_b_p = next_batch(test_size, X_te_p, y_te_p, 0)
        p_len = crnn_utils.pad(X_te_b_p, y_te_b_p,n_p_inputs, n_p_outputs)

        test_loss = s.run(total_loss, feed_dict={X_r: X_te_b_r, y_r: y_te_b_r, 
                        X_p: X_te_b_p, y_p: y_te_b_p, is_training: False})


        if test_loss < best_loss:
            best_loss = test_loss
            save_path = saver.save(s, "crnn_sep_save/crnn_separate_best.ckpt")


        # evaluation
        if e % 10 == 0: # only run sometimes

            # evaluation metrics
            acc_tr_r, acc_te_r, log_tr_r, log_te_r = [], [], [], []
            acc_tr_p, acc_te_p, log_tr_p, log_te_p = [], [], [], []
            total_log_tr, total_log_te = [], []

            # time-step representations
            truth_tr, pred_tr = [], []
            truth_te, pred_te = [], []

            # note representations
            conv_truth_r_tr, conv_truth_p_tr, conv_pred_r_tr, \
                    conv_pred_p_tr = [], [], [], []
            conv_truth_r_te, conv_truth_p_te, conv_pred_r_te, \
                    conv_pred_p_te = [], [], [], []

            # evaluate in batches            

            # train
            for b in range(n_batches_tr):
                
                # rhythm
                X_tr_b_r, y_tr_b_r = next_batch(eval_batch_size, 
						X_tr_r, y_tr_r, b)
                rhy_len = crnn_utils.pad(X_tr_b_r, y_tr_b_r,
                        n_r_inputs, n_r_outputs)

                # pitch
                X_tr_b_p, y_tr_b_p = next_batch(eval_batch_size, 
						X_tr_p, y_tr_p, b)
                p_len = crnn_utils.pad(X_tr_b_p, y_tr_b_p, 
                        n_p_inputs, n_p_outputs)

                # calc batch size (because of variable length final batch)
                curr_bs = len(X_tr_b_p)
                
                # evaluate training batch
                acc_p, log_p, out_p, acc_r, log_r, out_r, t_loss = \
                        s.run([accuracy_p, loss_p, logits_p, 
                        accuracy_r, loss_r, logits_r, total_loss], 
                        feed_dict={X_r: X_tr_b_r, y_r: y_tr_b_r, 
                        X_p: X_tr_b_p, y_p: y_tr_b_p, is_training: False})

                # add scores to lists
                # weighted by batch size to account for variable batch size
                acc_tr_r.append(acc_r * curr_bs)
                log_tr_r.append(log_r * curr_bs)
                acc_tr_p.append(acc_p * curr_bs)
                log_tr_p.append(log_p * curr_bs)
                total_log_tr.append(t_loss * curr_bs)

                # do conversions (out to note representation, 
                # out to time-steps representation) 
                conv_truth_r, conv_truth_p, conv_pred_r, conv_pred_p = \
                        out2notes(y_tr_b_r, y_tr_b_p, out_r, out_p)

                # add representations to lists
                conv_truth_r_tr += conv_truth_r
                conv_truth_p_tr += conv_truth_p
                conv_pred_r_tr += conv_pred_r
                conv_pred_p_tr += conv_pred_p

      
            # test
            for b in range(n_batches_te):

                # rhythm
                X_te_b_r, y_te_b_r = next_batch(eval_batch_size, 
						X_te_r, y_te_r, b)
                rhy_len = crnn_utils.pad(X_te_b_r, y_te_b_r, 
                        n_r_inputs, n_r_outputs)

                # pitch
                X_te_b_p, y_te_b_p = next_batch(eval_batch_size, 
						X_te_p, y_te_p, b)
                p_len = crnn_utils.pad(X_te_b_p, y_te_b_p, 
                        n_p_inputs, n_p_outputs)

                # calc batch size (because of variable length final batch)
                curr_bs = len(X_te_b_p)
                
                # evaluate testing batch
                acc_r, log_r, out_r, acc_p, log_p, out_p, loss = \
                        s.run([accuracy_r, loss_r, logits_r,
                        accuracy_p, loss_p, logits_p, total_loss], 
                        feed_dict={X_r: X_te_b_r, y_r: y_te_b_r, 
                        X_p: X_te_b_p, y_p: y_te_b_p,is_training: False})

                # add scores to lists
                # weighted by batch size to account for variable batch size
                acc_te_r.append(acc_r * curr_bs)
                log_te_r.append(log_r * curr_bs)
                acc_te_p.append(acc_p * curr_bs)
                log_te_p.append(log_p * curr_bs)
                total_log_te.append(loss * curr_bs)
		
                # do conversions (out to note representation, 
                # out to time-steps representation) 
                conv_truth_r, conv_truth_p, conv_pred_r, conv_pred_p = \
                        out2notes(y_te_b_r, y_te_b_p, out_r, out_p)


                # add representations to lists
                conv_truth_r_te += conv_truth_r
                conv_truth_p_te += conv_truth_p
                conv_pred_r_te += conv_pred_r
                conv_pred_p_te += conv_pred_p


            # compute total accuracy
            total_acc_tr = total_acc(conv_truth_r_tr, conv_truth_p_tr,
                    conv_pred_r_tr, conv_pred_p_tr)
            total_acc_te = total_acc(conv_truth_r_te, conv_truth_p_te, 
                    conv_pred_r_te, conv_pred_p_te)

            
            # print results
            print('\n\n------------ %d ------------' % e)
            print('...rhythm...')
            print('logp -tune', (sum(log_tr_r) / train_size), 
                    '-test', (sum(log_te_r)) / test_size)
            print('acc  -train', (sum(acc_tr_r) / train_size), 
                    '-tune', (sum(acc_te_r)) / test_size)

            print('\n...pitch...')
            print('logp -train', (sum(log_tr_p) / train_size), 
                    '-tune', (sum(log_te_p)) / test_size)
            print('acc  -train', (sum(acc_tr_p) / train_size), 
                    '-tune', (sum(acc_te_p)) / test_size)

            print('\n...combined...')
            print('total loss -train', (sum(total_log_tr) / train_size), 
                    '-tune', (sum(total_log_te)) / test_size)
            print('total acc  -train', total_acc_tr, '-tune', total_acc_te)

        else:
            # evaluate tuning set loss every epoch
            print("\nepoch:", e, "-total tuning loss:", test_loss)



print("\nBest tune loss: ", best_loss)



with tf.Session() as s:
        saver.restore(s, "crnn_sep_save/crnn_separate_best.ckpt")

        conv_truth_r_tr, conv_truth_p_tr, conv_pred_r_tr, conv_pred_p_tr = \
				[], [], [], []
        total_log_tr = []

        for b in range(n_batches_tr):
                    
            # rhythm
            X_tr_b_r, y_tr_b_r = next_batch(eval_batch_size, X_tr_r, y_tr_r, b)
            rhy_len = crnn_utils.pad(X_tr_b_r, y_tr_b_r,
                            n_r_inputs, n_r_outputs)

            # pitch
            X_tr_b_p, y_tr_b_p = next_batch(eval_batch_size, X_tr_p, y_tr_p, b)
            p_len = crnn_utils.pad(X_tr_b_p, y_tr_b_p, 
                            n_p_inputs, n_p_outputs)

            # calc batch size (because of variable length final batch)
            curr_bs = len(X_tr_b_p)
                    
            # evaluate training batch
            out_p, out_r, t_loss = s.run([logits_p,logits_r, total_loss], 
                            feed_dict={X_r: X_tr_b_r, y_r: y_tr_b_r, 
                            X_p: X_tr_b_p, y_p: y_tr_b_p, is_training: False})

            conv_truth_r, conv_truth_p, conv_pred_r, conv_pred_p = \
                     out2notes(y_tr_b_r, y_tr_b_p, out_r, out_p)

            conv_truth_r_tr += conv_truth_r
            conv_truth_p_tr += conv_truth_p
            conv_pred_r_tr += conv_pred_r
            conv_pred_p_tr += conv_pred_p

            total_log_tr.append(t_loss * curr_bs)


        #total accuracy
        total_acc_tr = total_acc(conv_truth_r_tr, conv_truth_p_tr,
                    conv_pred_r_tr, conv_pred_p_tr)



        #FINAL TEST
        rhy_len = crnn_utils.pad(X_r_final, y_r_final,
                            n_r_inputs, n_r_outputs)

        p_len = crnn_utils.pad(X_p_final, y_p_final,
                            n_p_inputs, n_p_outputs)


        out_p, out_r, t_loss_final = s.run([logits_p,logits_r, total_loss], 
                            feed_dict={X_r: X_r_final, y_r: y_r_final, 
                            X_p: X_p_final, y_p: y_p_final, is_training: False})

        conv_truth_r_f, conv_truth_p_f, conv_pred_r_f, conv_pred_p_f = \
                     out2notes(y_r_final, y_p_final, out_r, out_p)

        total_acc_final_te = total_acc(conv_truth_r_f, conv_truth_p_f,
                    conv_pred_r_f, conv_pred_p_f)


        print("\n\nFINAL RESULTS")
        print('\ntotal loss -train', (sum(total_log_tr) / train_size), 
                    '-final test', t_loss_final)
        print('total acc  -train', total_acc_tr, 
				'-final test', total_acc_final_te)

