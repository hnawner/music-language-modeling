#!/usr/bin/env python

from snn_rhythm import SNN
import tensorflow as tf

def tune():
    model = SNN(8, "/home/aeldrid2/REU/mels", "/home/aeldrid2/REU/mels/folk_minor", "both")

    model.construct([128, 128], tf.nn.tanh, 
        tf.nn.sparse_softmax_cross_entropy_with_logits)

    model.execute(128, 128)


tune()
