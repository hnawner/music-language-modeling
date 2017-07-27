#!/usr/bin/env python

import utils
import bigram_combined as bigram
import unigram_combined as unigram
import sys
    
def run_test(model, train, test):
    train_pitch, train_rhythm = utils.read_files(train, True)
    test_pitch, test_rhythm = utils.read_files(test, True)

    p_distr = model.distribution(train_pitch)
    r_distr = model.distribution(train_rhythm)
    print("-Train:-")
    model.neg_log_prob(train_pitch, train_rhythm, p_distr, r_distr)
    model.predict(train_pitch, train_rhythm, p_distr, r_distr)
    print("\n")
    print("-Test:-")
    model.neg_log_prob(test_pitch, test_rhythm, p_distr, r_distr)
    model.predict(test_pitch, test_rhythm, p_distr, r_distr)
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
    print("__Major__")
    run_test(bigram, train, test)

    print("Done.")
    return 0

if __name__ == '__main__':
    main()
