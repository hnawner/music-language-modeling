#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import utils
from utils import loss_fn, accuracy_fn
from utils import out2notes, total_acc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from copy import deepcopy
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import fully_connected

class RNN:

              
    def __init__(self, train, test, model_type, trans = True):

        self.train_path = train
        self.test_path = test
        self.trans = trans
        self.model_type = model_type
        
        self.start = 1 # always 1 for rnn model
        self.save_folder = "rnn_"+str(model_type)+"/best.ckpt"

        # training data
        # X = input, y = target
        # tr = train, te = test
        # r = rhythm, p = pitch
        (self.X_tr_r, self.X_te_r, self.y_tr_r, self.y_te_r, 
                self.X_tr_p, self.X_te_p, self.y_tr_p, self.y_te_p), \
                (self.p_max, self.p_min) = \
                utils.setup_rnn(self.train_path, self.start, self.trans)

        # data set sizes
        self.train_size = len(self.X_tr_r)
        self.test_size = len(self.X_te_r)


                
        self.build()


    def build(self):
        # p = pitch, r = rhythm
        self.n_p_inputs = len(self.X_tr_p[0][0])
        self.n_p_outputs = len(self.y_tr_p[0][0])
        self.n_r_inputs = len(self.X_tr_r[0][0])
        self.n_r_outputs = len(self.y_tr_r[0][0])
        
        
        self.X_p = tf.placeholder(tf.float32,
                [None, None, self.n_p_inputs], name="X_p")
        self.y_p = tf.placeholder(tf.float32, 
                [None, None, self.n_p_outputs], name="y_p")
        
        self.X_r = tf.placeholder(tf.float32, 
                [None, None, self.n_r_inputs], name="X_r")
        self.y_r = tf.placeholder(tf.float32, 
                [None, None, self.n_r_outputs], name="Y_r")
        
        if self.model_type == "combine":
            # concat pitch and rhythm
            combined = tf.concat([self.X_p, self.X_r], axis = 2)
        
            # RNN
            nu_rnn = 64
        
            cells = []
            cells.append(tf.contrib.rnn.LSTMCell(nu_rnn, use_peepholes = True, 
                    activation=tf.tanh))
            cells.append(tf.contrib.rnn.LSTMCell(nu_rnn,  use_peepholes = True, 
                    activation=tf.tanh))
            multi = tf.contrib.rnn.MultiRNNCell(cells)
            outs, _ = tf.nn.dynamic_rnn(multi, combined, dtype=tf.float32,
                    swap_memory=True, scope = "rhythm")
            outs_r = outs
            outs_p = outs

        elif self.model_type == "separate":
            # RNN
            nu_rnn = 64

            # RNN pitch
            cells_p = []
            cells_p.append(tf.contrib.rnn.LSTMCell(nu_rnn, 
                    use_peepholes = True, activation=tf.tanh))
            cells_p.append(tf.contrib.rnn.LSTMCell(nu_rnn, 
                    use_peepholes = True, activation=tf.tanh))
            multi_p = tf.contrib.rnn.MultiRNNCell(cells_p)
            outs_p, _ = tf.nn.dynamic_rnn(multi_p, self.X_p, 
                    dtype=tf.float32, swap_memory=True, scope = "pitch")

            # RNN rhythm
            cells_r = []
            cells_r.append(tf.contrib.rnn.LSTMCell(nu_rnn, 
                    use_peepholes = True, activation=tf.tanh))
            cells_r.append(tf.contrib.rnn.LSTMCell(nu_rnn, 
                    use_peepholes = True, activation=tf.tanh))
            multi_r = tf.contrib.rnn.MultiRNNCell(cells_r)
            outs_r, _ = tf.nn.dynamic_rnn(multi_r, self.X_r, 
                    dtype=tf.float32,swap_memory=True, scope = "rhythm")

        
        
        # batch normalization parameters
        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        
        bn_params = {"is_training": self.is_training, "decay": 0.999,
                "updates_collections":None, "scale":True}
        bn_params_out = {"is_training": self.is_training,
                "decay": 0.999, "updates_collections":None}
        
        # fully connected pitch
        n_hidden_1_p = 48
        n_hidden_2_p = 32
        keep_prob_p = 0.6
        
        stacked_outs_p = tf.reshape(outs_p, [-1, nu_rnn], name='stacked_outs_p')
        
        stacked_logits_p1 = fully_connected(stacked_outs_p, 
                    num_outputs = n_hidden_1_p,
                    activation_fn = tf.nn.elu,
                    normalizer_fn = batch_norm, 
                    normalizer_params=bn_params, 
                    scope='dense_p1')
        
        p_drop1 = tf.contrib.layers.dropout(stacked_logits_p1, 
                keep_prob_p, 
                is_training = self.is_training)
        
        stacked_logits_p2 = fully_connected(p_drop1, 
                    num_outputs = n_hidden_2_p, 
                    activation_fn = tf.nn.elu,
                    normalizer_fn = batch_norm, 
                    normalizer_params=bn_params, 
                    scope='dense_p2')
        
        p_drop2 = tf.contrib.layers.dropout(stacked_logits_p2, 
                keep_prob_p, 
                is_training = self.is_training)
        
        stacked_logits_p3 = fully_connected(p_drop2, 
                    num_outputs = self.n_p_outputs,
                    activation_fn = None,
                    normalizer_fn = batch_norm, 
                    normalizer_params=bn_params_out, 
                    scope='dense_p_out')
        
        self.logits_p = tf.reshape(stacked_logits_p3,
                [-1, tf.shape(self.y_p)[1], self.n_p_outputs],name='logits_p')
        
        # fully connected rhythm
        n_hidden_1_r = 48
        n_hidden_2_r = 32
        keep_prob_r = 0.6
        
        # separate rhythm
        stacked_outs_r = tf.reshape(outs_r, [-1, nu_rnn], name='stacked_outs_r')
        
        stacked_logits_r1 = fully_connected(stacked_outs_r, 
                    n_hidden_1_r, 
                    activation_fn = tf.nn.elu,
                    normalizer_fn = batch_norm, 
                    normalizer_params=bn_params, 
                    scope='dense_r1')
        
        r_drop1 = tf.contrib.layers.dropout(stacked_logits_r1, 
                keep_prob_r, 
                is_training = self.is_training)
        
        stacked_logits_r2 = fully_connected(r_drop1, 
                    n_hidden_2_r, 
                    activation_fn = tf.nn.elu,
                    normalizer_fn = batch_norm, 
                    normalizer_params=bn_params, 
                    scope='dense_r2')
        
        r_drop2 = tf.contrib.layers.dropout(stacked_logits_r2, 
                keep_prob_r, 
                is_training = self.is_training)
        
        stacked_logits_r3 = fully_connected(r_drop2, 
                    self.n_r_outputs, 
                    activation_fn = None,
                    normalizer_fn = batch_norm, 
                    normalizer_params=bn_params_out, 
                    scope='dense_r_out')
        
        self.logits_r = tf.reshape(stacked_logits_r3, 
                [-1, tf.shape(self.y_r)[1], self.n_r_outputs], name='logits_r')
        
        
        # loss params
        learn_rate = 0.02
        clip = 5
        
        # loss
        self.loss_r = loss_fn(self.logits_r, self.y_r)
        self.loss_p = loss_fn(self.logits_p, self.y_p)
        self.total_loss = tf.add(self.loss_r, self.loss_p)
        
        # training op
        if self.model_type == "combine":

            optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
            gradients = optimizer.compute_gradients(self.total_loss)
            capped_gradients = [(tf.clip_by_norm(grad, clip), var)
                    if grad != None else (grad, var) 
                    for grad, var in gradients]
            self.train_op = optimizer.apply_gradients(capped_gradients)


        elif self.model_type == "separate":

            optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    
            # rhythm
            gradients_r = optimizer.compute_gradients(self.loss_r)
            capped_gradients_r = [(tf.clip_by_norm(grad, clip), var) 
                    if grad != None else (grad, var) 
                    for grad, var in gradients_r]
            self.train_op_r = optimizer.apply_gradients(capped_gradients_r)
    
            # pitch
            gradients_p = optimizer.compute_gradients(self.loss_p)
            capped_gradients_p = [(tf.clip_by_norm(grad, clip), var) 
                    if grad != None else (grad, var) 
                    for grad, var in gradients_p]
            self.train_op_p = optimizer.apply_gradients(capped_gradients_p)
        
        # evaluation
        self.accuracy_r = accuracy_fn(self.logits_r, self.y_r)
        self.accuracy_p = accuracy_fn(self.logits_p, self.y_p)
        
        
        self.execute()



    # returns next batch
    def next_batch(self, size, x, y, n):
        start = size * n
        end = start + size
        if end >= len(x):
            return x[start:], y[start:] 
        return x[start:end], y[start:end]


    def execute(self):
        # execution params
        n_epochs = 100
        batch_size = 64
        self.eval_batch_size = 512 # larger to improve speed
        n_batches = int(np.ceil(len(self.X_tr_r) / batch_size))
        self.n_batches_tr = int(np.ceil(len(self.X_tr_r) / 
            self.eval_batch_size))
        self.n_batches_te = int(np.ceil(len(self.X_te_r) / 
            self.eval_batch_size))
        
        
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        
        # run session
        with tf.Session() as s:
        
            init.run()
        
            best_loss = sys.maxint
        
            for e in range(n_epochs):
        
                # shuffle training set each epoch
                X_shuf_p, y_shuf_p, X_shuf_r, y_shuf_r = shuffle(self.X_tr_p, 
                        self.y_tr_p, self.X_tr_r, self.y_tr_r)
        
                for b in range(n_batches):
        
                    # rhythm
                    X_batch_r, y_batch_r = self.next_batch(batch_size, 
                            X_shuf_r, y_shuf_r, b)
                    rhy_len = utils.pad(X_batch_r, y_batch_r, 
                            self.n_r_inputs, self.n_r_outputs)
        
                    # pitch
                    X_batch_p, y_batch_p = self.next_batch(batch_size,
                            X_shuf_p, y_shuf_p, b)
                    p_len = utils.pad(X_batch_p, y_batch_p, 
                            self.n_p_inputs, self.n_p_outputs)
        
                    # training op
                    if self.model_type == "combine":

                        s.run(self.train_op, feed_dict={self.X_p: X_batch_p, 
                            self.y_p: y_batch_p,self.X_r: X_batch_r, 
                            self.y_r: y_batch_r, self.is_training: True})

                    elif self.model_type == "separate":

                        s.run([self.train_op_r, self.train_op_p],
                             feed_dict={self.X_p: X_batch_p, 
                             self.y_p: y_batch_p,self.X_r: X_batch_r, 
                             self.y_r: y_batch_r, self.is_training: True})
        
               
                # evaluate total loss every epoch
                # rhythm
                X_te_b_r, y_te_b_r = self.next_batch(self.test_size, 
                        self.X_te_r, self.y_te_r, 0)
                rhy_len = utils.pad(X_te_b_r, y_te_b_r, self.n_r_inputs, 
                        self.n_r_outputs)
        
                # pitch
                X_te_b_p, y_te_b_p = self.next_batch(self.test_size, 
                        self.X_te_p,self.y_te_p, 0)
                p_len = utils.pad(X_te_b_p, y_te_b_p, self.n_p_inputs, 
                        self.n_p_outputs)
        
                test_loss = s.run(self.total_loss, 
                        feed_dict={self.X_r: X_te_b_r, 
                        self.y_r: y_te_b_r, self.X_p: X_te_b_p, 
                        self.y_p: y_te_b_p, self.is_training: False})
                
        
                if test_loss < best_loss:
                    best_loss = test_loss
                    save_path = self.saver.save(s, self.save_folder)
            
        
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
                    for b in range(self.n_batches_tr):
                        
                        # rhythm
                        X_tr_b_r, y_tr_b_r = self.next_batch(
                                self.eval_batch_size,self.X_tr_r, 
                                self.y_tr_r, b)
                        rhy_len = utils.pad(X_tr_b_r, y_tr_b_r,
                                self.n_r_inputs, self.n_r_outputs)
        
                        # pitch
                        X_tr_b_p, y_tr_b_p = self.next_batch(
                                self.eval_batch_size, 
                                self.X_tr_p, self.y_tr_p, b)
                        p_len = utils.pad(X_tr_b_p, y_tr_b_p, 
                                self.n_p_inputs, self.n_p_outputs)
        
                        # calc batch size (because of variable length batches)
                        curr_bs = len(X_tr_b_p)
                        
                        # evaluate training batch
                        acc_p, log_p, out_p, acc_r, log_r, out_r, t_loss = \
                                s.run([self.accuracy_p, self.loss_p, 
                                self.logits_p, self.accuracy_r, self.loss_r, 
                                self.logits_r, self.total_loss], 
                                feed_dict={self.X_r: X_tr_b_r, 
                                self.y_r: y_tr_b_r, self.X_p: X_tr_b_p, 
                                self.y_p: y_tr_b_p, self.is_training: False})
        
                        # add scores to lists
                        # weighted by batch size to account for variable size
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
                    for b in range(self.n_batches_te):
        
                        # rhythm
                        X_te_b_r, y_te_b_r = self.next_batch(
                                self.eval_batch_size, 
                                self.X_te_r, self.y_te_r, b)
                        rhy_len = utils.pad(X_te_b_r, y_te_b_r, 
                                self.n_r_inputs, self.n_r_outputs)
        
                        # pitch
                        X_te_b_p, y_te_b_p = self.next_batch(
                                self.eval_batch_size, 
                                self.X_te_p, self.y_te_p, b)
                        p_len = utils.pad(X_te_b_p, y_te_b_p, 
                                self.n_p_inputs, self.n_p_outputs)
        
                        # calc batch size (because of variable length batches)
                        curr_bs = len(X_te_b_p)
                        
                        # evaluate testing batch
                        acc_r, log_r, out_r, acc_p, log_p, out_p, loss = \
                                s.run([self.accuracy_r, self.loss_r, 
                                self.logits_r, self.accuracy_p, self.loss_p, 
                                self.logits_p, self.total_loss], 
                                feed_dict={self.X_r: X_te_b_r, 
                                self.y_r: y_te_b_r,self.X_p: X_te_b_p,
                                self.y_p: y_te_b_p, 
                                self.is_training: False})
        
                        # add scores to lists
                        # weighted by batch size to account for variable size
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
                    total_acc_tr = total_acc(conv_truth_r_tr, 
                            conv_truth_p_tr,conv_pred_r_tr, conv_pred_p_tr)
                    total_acc_te = total_acc(conv_truth_r_te, 
                            conv_truth_p_te, conv_pred_r_te, conv_pred_p_te)
        
                    
                    # print results
                    print('\n------------ %d ------------' % e)
                    print('...rhythm...')
                    print('loss -train', (sum(log_tr_r) / self.train_size), 
                            '-tune', (sum(log_te_r)) / self.test_size)
                    print('acc  -train', (sum(acc_tr_r) / self.train_size), 
                            '-tune', (sum(acc_te_r)) / self.test_size)
        
                    print('\n...pitch...')
                    print('loss -train', (sum(log_tr_p) / self.train_size), 
                            '-tune', (sum(log_te_p)) / self.test_size)
                    print('acc  -train', (sum(acc_tr_p) / self.train_size), 
                            '-tune', (sum(acc_te_p)) / self.test_size)
        
                    print('\n...combined...')
                    print('total loss -train', (sum(total_log_tr) / 
                            self.train_size), '-tune', (sum(total_log_te)) / 
                            self.test_size)
                    print('total acc  -train', total_acc_tr, 
                            '-tune', total_acc_te)
        
                else:
                    # evaluate tuning set loss every epoch
                    print("\nepoch:", e, "-total tuning loss:", test_loss)
                
                
        
        print("\nBest test loss: ", best_loss)
        
        if self.test_path != None:
            self.final_test()

    def final_test(self):

        # final test data
        self.X_p_final, self.y_p_final, self.X_r_final, self.y_r_final = \
                utils.setup_test(self.test_path, self.p_max, 
                                    self.p_min, self.start, self.trans)
        with tf.Session() as s:
            self.saver.restore(s, self.save_folder)
    
            conv_truth_r_tr, conv_truth_p_tr, conv_pred_r_tr, \
                    conv_pred_p_tr = [], [], [], []
            total_log_tr = []
    
            for b in range(self.n_batches_tr):
                        
                # rhythm
                X_tr_b_r, y_tr_b_r = self.next_batch(self.eval_batch_size, 
                        self.X_tr_r, self.y_tr_r, b)
                rhy_len = utils.pad(X_tr_b_r, y_tr_b_r,
                                self.n_r_inputs, self.n_r_outputs)
    
                # pitch
                X_tr_b_p, y_tr_b_p = self.next_batch(self.eval_batch_size, 
                        self.X_tr_p, self.y_tr_p, b)
                p_len = utils.pad(X_tr_b_p, y_tr_b_p, 
                                self.n_p_inputs, self.n_p_outputs)
    
                # calc batch size (because of variable length final batch)
                curr_bs = len(X_tr_b_p)
                        
                # evaluate training batch
                out_p, out_r, t_loss = s.run([self.logits_p,self.logits_r, 
                        self.total_loss], 
                        feed_dict={self.X_r: X_tr_b_r, self.y_r: y_tr_b_r, 
                        self.X_p: X_tr_b_p, self.y_p: y_tr_b_p, 
                        self.is_training: False})
    
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
            rhy_len = utils.pad(self.X_r_final, self.y_r_final,
                                self.n_r_inputs, self.n_r_outputs)
    
            p_len = utils.pad(self.X_p_final, self.y_p_final,
                                self.n_p_inputs, self.n_p_outputs)
    
    
            out_p, out_r, t_loss_final = s.run([self.logits_p, self.logits_r, 
                        self.total_loss], 
                        feed_dict={self.X_r: self.X_r_final, 
                        self.y_r: self.y_r_final, 
                        self.X_p: self.X_p_final, 
                        self.y_p: self.y_p_final, 
                        self.is_training: False})
    
            conv_truth_r_f, conv_truth_p_f, conv_pred_r_f, conv_pred_p_f = \
                         out2notes(self.y_r_final, self.y_p_final, 
                         out_r, out_p)
    
            total_acc_final_te = total_acc(conv_truth_r_f, conv_truth_p_f,
                        conv_pred_r_f, conv_pred_p_f)
    
    
            print("\n\nFINAL RESULTS")
            print('\ntotal loss -train', (sum(total_log_tr) / self.train_size), 
                        '-final test', t_loss_final)
            print('total acc  -train', total_acc_tr, 
                '-final test', total_acc_final_te)


