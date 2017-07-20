#!/usr/bin/env python

from cnn import CNN

n = CNN(7, 'mels/folk_major', 'major')
n.construct()
n.execute(32, 100)
