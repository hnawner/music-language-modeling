#!/usr/bin/env python

# SNN test using trigrams - 8-grams
# tanh activation function, cross-entropy loss function

import sys
import tensorflow as tf
from snn import SequentialNN as SNN

def main():
    
    i = int(sys.argv[1])

    layers = [(256,256), (512,512,512)]
    path = 'mels/folk'
    
    mode = 'major' if i < 12 else 'minor'
    ngramLen = (i % 6) + 3
    arch = layers[(i // 6) % 2]
    actFun = tf.tanh
    lossFun = tf.nn.sparse_softmax_cross_entropy_with_logits
    batchsize = 512
    epochs = 256

    print("Test #" + str(i))
    print("Architecture:", arch)
    print("nGram:", ngramLen, '\n')

    snn = SNN(ngramLen, path, mode)
    snn.construct(arch, actFun, lossFun)
    snn.execute(batchsize, epochs)

if __name__ == '__main__':
    main()
