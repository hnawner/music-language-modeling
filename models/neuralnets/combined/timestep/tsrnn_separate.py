#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
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

## PITCH AND RHTYHM INTERACT

print("Start")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))



## HELPER AND EVALUATION FUNCTIONS ##

# softmax cross entropy
def loss_fn(logits, Y, weight_loss, training):
    xentropy = Y * tf.log(tf.nn.softmax(logits))
    if training:
        xentropy *= weight_loss
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


# translates neural net outputs and targets from note representation
# to time step representation with subdivisions of "96th" note
def notes2timesteps(truth_r, truth_p, pred_r, pred_p):
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
            t_val = ([k for k, v in nrnn_utils.r_dict.items() 
                    if v == (np.argmax(t_r[step]))])[0]
            t_len = int(round((t_val * scale) / subdiv))
            t_pitch = np.argmax(t_p[step])
            new_t_ts = [t_pitch] + ([0] * t_len)
            t += new_t_ts

            p_val = ([k for k, v in nrnn_utils.r_dict.items() 
                    if v == (np.argmax(p_r[step]))])[0]
            p_len = int(round((p_val * scale) / subdiv))
            p_pitch = np.argmax(p_p[step])
            new_p_ts = [p_pitch] + ([0] * p_len)
            p += new_p_ts        
            
        truth.append(t)
        pred.append(p)	

    return truth, pred


# computes f1 score from time-steps representation
def ts_f1(preds, labels):

    false_pos = false_neg = true_pos = 0

    for p, l in zip(preds, labels):
        
        # pad so pitch and rhythm sequences are equal length
        if len(p) > len(l): 
            diff = len(p) - len(l)
            l += [0] * diff
        elif len(l) > len(p):
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

    # compute f1
    if true_pos == 0:
        return 0
    prec =  float(true_pos) / (true_pos + false_pos)
    recall = float(true_pos) / (true_pos + false_neg)
    f1 = (2 * (prec * recall) / (prec + recall))

    return f1


# converts the outputs of the neural net and the targets 
# from one-hot vectors to note representation
def out2ts(truth, pred):
    conv_truth = []
    conv_pred = []

    for t, p in zip(truth, pred):
        t = [np.argmax(i) for i in t if sum(i) != 0]
        p = ([np.argmax(i) for i in p])[:len(t)]


        conv_truth.append(t)
        conv_pred.append(p)

    return conv_truth, conv_pred

# converts time steps representation to note representation
def ts2notes(truth, pred, step):
    conv_truth_p = []
    conv_truth_r = []
    conv_pred_p = []
    conv_pred_r = []

    # convert truth
    for t in truth:
        pitches = []
        rhythms = []
        
        curr_len = 0
        curr_pitch = None
        for i in range(len(t)):
            if curr_len != 0:
                if t[i] != 0: # start of note
                    prev_len = curr_len * step
                    rhythms.append(prev_len)
                    pitches.append(curr_pitch)
                    curr_pitch = t[i]
                    curr_note = 1
                    
                elif t[i] == 0: #no event
                    curr_len += 1
            elif pitches == []:
                if t[i] != 0: # start of note
                    curr_len += 1  
                    curr_pitch = t[i]
                elif t[i] == 0: #no event
                    curr_len += 1

        conv_truth_p.append(pitches)
        conv_truth_r.append(rhythms)

    # convert predictions
    for p in pred:
        pitches = []
        rhythms = []
        
        curr_len = 0
        curr_pitch = None
        for i in range(len(p)):
            if curr_len != 0:
                if p[i] != 0: # start of note
                    prev_len = curr_len * step
                    rhythms.append(prev_len)
                    pitches.append(curr_pitch)
                    curr_pitch = p[i]
                    curr_note = 1
                    
                elif p[i] == 0: #no event
                    curr_len += 1
            elif pitches == []:
                if p[i] != 0: # start of note
                    curr_len += 1  
                    curr_pitch = p[i]
                elif p[i] == 0: #no event
                    curr_len += 1

        conv_pred_p.append(pitches)
        conv_pred_r.append(rhythms)

    return conv_truth_r, conv_truth_p, conv_pred_r, conv_pred_p


# computes f1 of notes representation in the best alignment
# between prediction and target
def aligned_f1(truth_r, truth_p, pred_r, pred_p):

    tp = 0
    fp = 0
    fn = 0

    for t_r, t_p, p_r, p_p in zip(truth_r, truth_p, pred_r, pred_p):

        # creat dynamic programming table
        compare = np.array([[0] * (len(p_r) + 1)] * (len(t_r) + 1))

        # initialize edges
        '''
        for i in range(1, len(compare[0])):
            compare[0, i] = i * (-1)
        for i in range(1, len(compare)):
            compare[i, 0] = i * (-1)
        '''

        # fill table
        for t in range(1, len(compare)):
            for p in range(1, len(compare[0])):
                match = (t_r[t-1] == p_r[p-1]) and (t_p[t-1] == p_p[p-1])
                if match: match = 1
                else: match = 0
                m = compare[t-1,p-1] + match
                gapt = compare[t-1,p]
                gapp = compare[t,p-1]

                compare[t,p] = max(m, max(gapt, gapp))
        # retrieve score of best alignment
        correct = np.amax(compare)

        tp += correct   
        fn += (len(t_p) - correct)
        fp += (len(p_p) - correct)

    # compute f1
    if tp == 0:
        return 0
    prec =  float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = (2 * (prec * recall) / (prec + recall))

    return f1


# computes accuracy predictions in note representation
def total_acc(truth_r, truth_p, pred_r, pred_p):
    total = 0
    correct = 0
    for t_r, t_p, p_r, p_p in zip(truth_r, truth_p, pred_r, pred_p):
        max_len = max(len(t_r), len(p_r))
        for i in range(max_len):
            if i < len(t_r) and i < len(p_r):
                if (t_r[i] == p_r[i]) and (t_p[i] == p_p[i]):
                    correct += 1
            total += 1

    acc= correct / total

    return acc            



## GET DATA ##

print("\nGet data")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

path = '/home/aeldrid2/REU/krn_split/converted/other/debug/'
TIME_STEP = 250
PITCH_MIN = 40
PITCH_MAX = 90
PITCH_RANGE = PITCH_MAX - PITCH_MIN
start = 16 # length of cnn window in notes


# X = input, y = target
# tr = train, te = test
(X_tr, X_te, y_tr, y_te) = crnn_utils.get_data(path, TIME_STEP, PITCH_MIN, PITCH_MAX, start)



# data set sizes
train_size = len(X_tr)
test_size = len(X_te)

X_len = len(X_tr[0][0])
y_len = len(y_tr[0][0])



## BUILD NETWORK ##

print("\nBuild networks")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

# p = pitch, r = rhythm
n_inputs = X_len
n_outputs = y_len


X = tf.placeholder(tf.float32, [None, None, n_inputs], name="X")
y = tf.placeholder(tf.float32, [None, None, n_outputs], name="y")




# CNN 
cnn = tf.layers.conv1d(inputs=X,
                           filters=16, 
                           kernel_size=8,
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv_1')

cnn = tf.layers.conv1d(inputs=cnn,
                           filters=16, 
                           kernel_size=9, 
                           padding='valid',
                           activation=tf.nn.relu,
                           name='conv_2')


# RNN
nu_rnn = 64

cells = []
cells.append(tf.contrib.rnn.LSTMCell(nu_rnn, use_peepholes = True, 
		activation=tf.tanh))
cells.append(tf.contrib.rnn.LSTMCell(nu_rnn,  use_peepholes = True, 
		activation=tf.tanh))
multi = tf.contrib.rnn.MultiRNNCell(cells)
outs, _ = tf.nn.dynamic_rnn(multi, cnn, dtype=tf.float32,
        swap_memory=True, scope = "rnn")


# batch normalization parameters
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

bn_params = {"is_training": is_training, "decay": 0.99,
        "updates_collections":None, "scale":True}
bn_params_out = {"is_training": is_training,
        "decay": 0.99, "updates_collections":None}

# fully connected
n_hidden_1 = 64
n_hidden_2 = 64

stacked_outs = tf.reshape(outs, [-1, nu_rnn], name='stacked_outs')

stacked_logits1 = fully_connected(stacked_outs, 
			num_outputs = n_hidden_1,
			activation_fn = tf.nn.elu,
			normalizer_fn = batch_norm, 
			normalizer_params=bn_params, 
			scope='dense_1')

stacked_logits2 = fully_connected(stacked_logits1, 
			num_outputs = n_hidden_2, 
			activation_fn = tf.nn.elu,
			normalizer_fn = batch_norm, 
			normalizer_params=bn_params, 
			scope='dense_2')

stacked_logits3 = fully_connected(stacked_logits2, 
			num_outputs = n_outputs,
			activation_fn = None,
			normalizer_fn = batch_norm, 
			normalizer_params=bn_params_out, 
			scope='dense_out')

logits = tf.reshape(stacked_logits3,
        [-1, tf.shape(y)[1], n_outputs],name='logits')


# loss params
learn_rate = 0.01
clip = 5
weight_loss = tf.placeholder(tf.float32, [None, None, n_outputs], name="weight_loss")


# loss
loss_train = loss_fn(logits, y, weight_loss, True)

# optimizer with gradient clipping
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
gradients = optimizer.compute_gradients(loss_train)
capped_gradients = [(tf.clip_by_norm(grad, clip), var) for grad, var in gradients]
train_op = optimizer.apply_gradients(capped_gradients)

# evaluation
loss = loss_fn(logits, y, None, False)
accuracy = accuracy_fn(logits, y)



## EXECUTION PHASE ##

print("\nExecute")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

# returns next batch
def next_batch(size, x, y, n):
    start = size * n
    end = start + size
    if end >= len(x):
        return x[start:], y[start:] 
    return x[start:end], y[start:end]

# execution params
n_epochs = 11
batch_size = 128
eval_batch_size = 512 # larger to improve speed
n_batches = int(np.ceil(len(X_tr) / batch_size))
n_batches_tr = int(np.ceil(len(X_tr) / eval_batch_size))
n_batches_te = int(np.ceil(len(X_te) / eval_batch_size))
no_event_weight = 0.2
other_weight = 2


init = tf.global_variables_initializer()
saver = tf.train.Saver()

# run session
with tf.Session() as s:

    init.run()

    best_loss = sys.maxint

    for e in range(n_epochs):

        print("\nTrain epoch ", e)
        print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

        # shuffle training set each epoch
        X_shuf, y_shuf = shuffle(X_tr, y_tr)

        for b in range(n_batches):

            # batch
            X_batch, y_batch = next_batch(batch_size, 
                    X_shuf, y_shuf, b)
            max_len = crnn_utils.pad(X_batch, y_batch, X_len, y_len)
            curr_bs = len(X_batch)

            loss_weights = [no_event_weight] + [other_weight] * (n_outputs -1)		
            loss_weight_mat = [[loss_weights] * (max_len - start + 1) ] * curr_bs


            # training op
            s.run(train_op, feed_dict={X: X_batch, y: y_batch, 
                    is_training: True, weight_loss: loss_weight_mat})


       
        # evaluate total loss every epoch
        # rhythm
        X_te_eval, y_te_eval = X_te, y_te
        rhy_len = crnn_utils.pad(X_te_eval, y_te_eval, X_len, y_len)

        test_loss = s.run(loss, feed_dict={X: X_te_eval, y: y_te_eval, 
                is_training: False})
        
        print("\nTotal test loss: ", test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            save_path = saver.save(s, "tsrnn_comb_save/tsrnn_combined_best.ckpt")
    

        # evaluation
        if e % 5 == 0: # only run sometimes

            print("\nEvaluate epoch ", e)
            print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

            # evaluation metrics
            acc_tr_ts, acc_te_ts = [], []
            loss_tr_ts, loss_te_ts = [], []

            # time-step representations
            truth_tr, pred_tr = [], []
            truth_te, pred_te = [], []

            # note representations
            conv_truth_r_tr, conv_truth_p_tr, conv_pred_r_tr, conv_pred_p_tr = [], [], [], []
            conv_truth_r_te, conv_truth_p_te, conv_pred_r_te, conv_pred_p_te = [], [], [], []

            # evaluate in batches            

            # train
            for b in range(n_batches_tr):
                
                # batch
                X_tr_b, y_tr_b = next_batch(eval_batch_size, X_tr, y_tr, b)
                rhy_len = crnn_utils.pad(X_tr_b, y_tr_b, X_len, y_len)

                # calc batch size (because of variable length final batch)
                curr_bs = len(X_tr_b)
                
                # evaluate training batch
                ts_acc, ts_loss, ts_logits = s.run([accuracy, loss, logits], 
                        feed_dict={X: X_tr_b, y: y_tr_b, is_training: False})

                # add scores to lists
                # weighted by batch size to account for variable batch size
                acc_tr_ts.append(ts_acc * curr_bs)
                loss_tr_ts.append(ts_loss * curr_bs)

                # do conversions (out to time_step representation, 
                # then time-steps to note representation) 
                truth, pred = out2ts(y_tr_b, ts_logits)
                conv_truth_r, conv_truth_p, conv_pred_r, conv_pred_p = \
                        ts2notes(truth, pred, TIME_STEP)

                # add representations to lists
                conv_truth_r_tr += conv_truth_r
                conv_truth_p_tr += conv_truth_p
                conv_pred_r_tr += conv_pred_r
                conv_pred_p_tr += conv_pred_p

                truth_tr += truth
                pred_tr += pred
      
            # test
            for b in range(n_batches_te):

                # batch
                X_te_b, y_te_b = next_batch(eval_batch_size, X_te, y_te, b)
                rhy_len = crnn_utils.pad(X_te_b, y_te_b, X_len, y_len)

                # calc batch size (because of variable length final batch)
                curr_bs = len(X_te_b)
                
                # evaluate training batch
                ts_acc, ts_loss, ts_logits = s.run([accuracy, loss, logits], 
                        feed_dict={X: X_te_b, y: y_te_b, is_training: False})

                # add scores to lists
                # weighted by batch size to account for variable batch size
                acc_te_ts.append(ts_acc * curr_bs)
                loss_te_ts.append(ts_loss * curr_bs)

                # do conversions (out to time_step representation, 
                # then time-steps to note representation) 
                truth, pred = out2ts(y_te_b, ts_logits)
                conv_truth_r, conv_truth_p, conv_pred_r, conv_pred_p = \
                        ts2notes(truth, pred, TIME_STEP)

                # add representations to lists
                conv_truth_r_te += conv_truth_r
                conv_truth_p_te += conv_truth_p
                conv_pred_r_te += conv_pred_r
                conv_pred_p_te += conv_pred_p

                truth_te += truth
                pred_te += pred


            # example predictions and targets for analysis
            print("truth 1",truth_te[0])
            print("pred 1",pred_te[0])

            print("\ntruth 1",truth_te[1])
            print("pred 1",pred_te[1])



            # compute overall scores

            print("\nCompute scores ", e)
            print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

            # time-step f1
            ts_f1_tr = ts_f1(truth_tr, pred_tr)
            ts_f1_te = ts_f1(truth_te, pred_te)

            # aligned f1
            aligned_f1_tr = aligned_f1(conv_truth_r_tr, conv_truth_p_tr, 
                    conv_pred_r_tr, conv_pred_p_tr)
            aligned_f1_te = aligned_f1(conv_truth_r_te, conv_truth_p_te, 
                    conv_pred_r_te, conv_pred_p_te)

            # note accuracy
            note_acc_tr = total_acc(conv_truth_r_tr, conv_truth_p_tr,
                    conv_pred_r_tr, conv_pred_p_tr)
            note_acc_te = total_acc(conv_truth_r_te, conv_truth_p_te, 
                    conv_pred_r_te, conv_pred_p_te)

            
            # print results
            print('\n\n------------ %d ------------' % e)
            print('\n...combined...')
            print('total loss -train', (sum(loss_tr_ts) / train_size), 
                    '-test', (sum(loss_te_ts)) / test_size)
            print('ts acc -train', (sum(acc_tr_ts) / train_size), 
                    '-test', (sum(acc_te_ts)) / test_size)
            print('note acc  -train', note_acc_tr, '-test', note_acc_te)
            print('\ntime-step f1  -train', ts_f1_tr, '-test', ts_f1_te)
            print('note aligned f1  -train', aligned_f1_tr, '-test', aligned_f1_te)
            print('\n')
        

print("\nBest test loss: ", best_loss)


print("\nDone")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
