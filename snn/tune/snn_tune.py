#!/usr/bin/env python

from snn import SequentialNN as snn
import tensorflow as tf
import utils
import sys

def main():
    i = int(sys.argv[1])
    m = "minor"
    d = "/home/aeldrid2/REU/mels/folk_minor"
    n = (i % 7) + 2
    if(i < 7):
        m = "major"
        d = "/home/aeldrid2/REU/mels/folk_major"
    learn = 0.25
    nEpochs = 256
    batchSize = 512
    arch = [256,512,256]
    e = utils.one_hot_ngram_PCandOctave
    inSize = 20
    lossfn = tf.nn.sparse_softmax_cross_entropy_with_logits
    a = tf.nn.tanh

    print("Mode: ", m)
    print("n: ", n)
    print("Architecture: ", arch)
    print("Encoder: ", e) 

    modelMaj = snn(n, d, m, encoder = e, inputSize = inSize)
    modelMaj.construct(architecture = arch, act = a, lossFun = lossfn, learningRate = learn)
    modelMaj.execute(batchSize, nEpochs)


main()
