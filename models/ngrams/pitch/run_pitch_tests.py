#!/usr/bin/env python

import utils
import bigram
import interval_bigram as intBg
import interval_unigram as intUg
import unigram
import sys
    
def run_test(model, train, test):
    train = utils.read_files(train)
    test= utils.read_files(test)
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
    if len(sys.argv) != 3:
        print("Usage: folder containing mel files")
        return 1

    train = sys.argv[1]
    test = sys.argv[2]
    
    print("*** Unigram Test ***")
    run_test(unigram, train, test)
   
    print("*** Bigram Test ***")
    run_test(bigram, train, test)
   
    print("*** Interval Unigram Test ***")
    run_test(intUg, train, test)

    print("*** Interval Bigram Test ***")
    run_test(intBg, train, test)


    print("Done.")
    return 0

if __name__ == '__main__':
    main()
