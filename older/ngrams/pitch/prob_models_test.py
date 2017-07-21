#!/usr/bin/env python

import utils
import bigram
import interval_bigram as intBg
import interval_unigram as intUg
import unigram
import sys
    
def run_test(model, train, test):
    distr = model.distribution(train)
    print("-Train:-")
    model.neg_log_prob(train, distr)
    model.predict(train, distr)
    print("\n")
    print("-Test:-")
    model.neg_log_prob(test, distr)
    model.predict(test, distr)
    print("\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: folder containing mel files")
        return 1

    path = sys.argv[1]

    # Read training files
    fmaj = path + "folk_major/"
    fmin = path + "folk_minor/"
    maj_train = utils.read_files(fmaj)
    min_train = utils.read_files(fmin)
    
    # Read test files
    fmajtest = path + "folk_maj_test/"
    fmintest = path + "folk_min_test/"
    maj_test = utils.read_files(fmajtest)
    min_test = utils.read_files(fmintest)
    
    print("*** Unigram Test ***")
    print("__Major__")
    run_test(unigram, maj_train, maj_test)
    print("__Minor__")
    run_test(unigram, min_train, min_test)
    
    print("*** Bigram Test ***")
    print("__Major__")
    run_test(bigram, maj_train, maj_test)
    print("__Minor__")
    run_test(bigram, min_train, min_test)
    
    print("*** Interval Unigram Test ***")
    print("__Major__")
    run_test(intUg, maj_train, maj_test)
    print("__Minor__")
    run_test(intUg, min_train, min_test)
    
    print("*** Interval Bigram Test ***")
    print("__Major__")
    run_test(intBg, maj_train, maj_test)
    print("__Minor__")
    run_test(intBg, min_train, min_test)

    print("Done.")
    return 0

if __name__ == '__main__':
    main()
