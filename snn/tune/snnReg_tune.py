#!/usr/bin/env python

from snnReg import SequentialNNReg as snnReg
import tensorflow as tf
import utils
import sys

def main():
    i = int(sys.argv[1])
    r = 0.0001 * i + 0.0001
    m = "major"
    d = "/home/aeldrid2/REU/mels/folk_major"
    e = utils.one_hot_ngram_PCandOctave
    inSize = 20
    arch = [256, 512, 256]
    a = tf.nn.tanh
    lossfn = tf.nn.sparse_softmax_cross_entropy_with_logits
    learn = 0.04
    nEpochs = 256
    batchSize = 512
    n = 8
    regfn = tf.contrib.layers.l1_regularizer
    fileName = "SNN_reg_l1_results.txt"


    details = "Test: " + str (i) + "\nMode: " +  str(m) + "\nn: " + str(n) + "\nArchitecture: " + str(arch) +"\nEncoder: " + str(e) + "\nReg Function: " + str(regfn) + "\nReg scale: " + str(r) +"\nLearning rate: " + str(learn)


    model = snnReg(details, n, d, m, encoder = e, inputSize = inSize)
    model.construct(architecture = arch, act = a, lossFun = lossfn, regFun = regfn, reg_scale = r, learningRate = learn)
    model.execute(batchSize, nEpochs, fileName)




