#!/usr/bin/env python

import utils
import bigram_no_condense as bigram
import interval_bigram as intBg
import interval_unigram as intUg
import unigram

# Read training files
maj_train, temp = utils.read_files("/home/aeldrid2/REU/mels/folk_major")
temp, min_train = utils.read_files("/home/aeldrid2/REU/mels/folk_minor")

# Read test files
maj_test, temp = utils.read_files("/home/aeldrid2/REU/mels/folk_maj_test")
temp, min_test = utils.read_files("/home/aeldrid2/REU/mels/folk_min_test")

print("____Unigram Test____")
maj_distr = unigram.distribution(maj_train)
min_distr = unigram.distribution(min_train)

print("__Major Train:__")
unigram.neg_log_prob(maj_train, maj_distr)
unigram.predict(maj_train, maj_distr)
print("\n")

print("__Minor Train:__")
unigram.neg_log_prob(min_train, min_distr)
unigram.predict(min_train, min_distr)
print("\n")

print("__Major Test:__")
unigram.neg_log_prob(maj_test, maj_distr)
unigram.predict(maj_test, maj_distr)
print("\n")

print("__Minor Test:__")
unigram.neg_log_prob(min_test, min_distr)
unigram.predict(min_test, min_distr)
print("\n")
print("\n")


print("____Bigram Test____")
maj_distr = bigram.distribution(maj_train)
min_distr = bigram.distribution(min_train)

print("__Major Train:__")
bigram.neg_log_prob(maj_train, maj_distr)
bigram.predict(maj_train, maj_distr)
print("\n")

print("__Minor Train:__")
bigram.neg_log_prob(min_train, min_distr)
bigram.predict(min_train, min_distr)
print("\n")

print("__Major Test:__")
bigram.neg_log_prob(maj_test, maj_distr)
bigram.predict(maj_test, maj_distr)
print("\n")

print("__Minor Test:__")
bigram.neg_log_prob(min_test, min_distr)
bigram.predict(min_test, min_distr)
print("\n")
print("\n")



print("____Interval Unigram Test____")
maj_distr = intUg.distribution(maj_train)
min_distr = intUg.distribution(min_train)

print("__Major Train:__")
intUg.neg_log_prob(maj_train, maj_distr)
intUg.predict(maj_train, maj_distr)
print("\n")

print("__Minor Train:__")
intUg.neg_log_prob(min_train, min_distr)
intUg.predict(min_train, min_distr)
print("\n")

print("__Major Test:__")
intUg.neg_log_prob(maj_test, maj_distr)
intUg.predict(maj_test, maj_distr)
print("\n")

print("__Minor Test:__")
intUg.neg_log_prob(min_test, min_distr)
intUg.predict(min_test, min_distr)
print("\n")
print("\n")


print("____Interval Bigram Test____")
maj_distr = intBg.distribution(maj_train)
min_distr = intBg.distribution(min_train)

print("__Major Train:__")
intBg.neg_log_prob(maj_train, maj_distr)
intBg.predict(maj_train, maj_distr)
print("\n")

print("__Minor Train:__")
intBg.neg_log_prob(min_train, min_distr)
intBg.predict(min_train, min_distr)
print("\n")

print("__Major Test:__")
intBg.neg_log_prob(maj_test, maj_distr)
intBg.predict(maj_test, maj_distr)
print("\n")

print("__Minor Test:__")
intBg.neg_log_prob(min_test, min_distr)
intBg.predict(min_test, min_distr)




