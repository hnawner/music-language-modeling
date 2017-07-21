#!/usr/bin/env python

from snnDrop import SequentialNNDrop as snn
import tensorflow as tf
import utils
import sys

def main():
    i = "other test"
    e = utils.one_hot_ngram_PCandOctave
    n = 8
    inSize = 20
    m = "minor"
    d = "/home/aeldrid2/REU/mels/folk_minor"
    k = 0.35
    #if(i < 7):
    #    m = "major"
    #    d = "/home/aeldrid2/REU/mels/folk_major"
    arch = [512, 1024, 1024, 512]
    learn = 0.25
    nEpoch = 256
    batchSize = 128
    lossfn = tf.nn.sparse_softmax_cross_entropy_with_logits
    a = tf.nn.tanh
    resultsFile = "SNN_drop_results.txt"

    details = "\nTest: " + str (i) + "\nMode: " +  str(m) + "\nn: " + str(n) + "\nArchitecture: " + str(arch) +"\nEncoder: " + str(e) + "\nKeep prob : " + str(k) +"\nLearning rate: " + str(learn) +"\n"

    modelMaj = snn(details, n, data = d, mode = m, encoder = e, inputSize = inSize)
    modelMaj.construct(architecture = arch, act = a, lossFun = lossfn, keep_prob = k, learningRate = learn)
    modelMaj.execute(batchSize, nEpoch, resultsFile)


main()
