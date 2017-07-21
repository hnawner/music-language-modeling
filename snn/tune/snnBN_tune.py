#!/usr/bin/env python

from snnBN import SequentialNNBN as snnBN
import tensorflow as tf
import utils
import sys

def main():
    i = int(sys.argv[1])
    m = "minor"
    d = "/home/aeldrid2/REU/mels/folk_minor"
    n = (i % 9) + 2
    if(i < 9):
        m = "major"
        d = "/home/aeldrid2/REU/mels/folk_major"
    nEpochs = 512
    batchSize = 128
    arch = [256,512,256]
    e = utils.one_hot_ngram_PCandOctave
    inSize = 20
    lossfn = tf.nn.sparse_softmax_cross_entropy_with_logits
    a = tf.nn.tanh
    k = 0.4
    learn = 0.01

    print("Mode: ", m)
    print("n: ", n)
    print("Architecture: ", arch)
    print("Encoder: ", e) 

    modelMaj = snnBN(n, d, m, encoder = e, inputSize = inSize)
    modelMaj.construct(architecture = arch, act = a, 
		lossFun = lossfn, keep_prob = k, learningRate = learn)
    modelMaj.execute(batchSize, nEpochs)


main()
