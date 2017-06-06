from __future__ import division
import os, sys
from math import log
# from functools import reduce

def distributions(folder):
        # list of files in input directory
    files = os.listdir(folder)
    num_files = len(files)

    # number of files used in training set
    train_max = int(.7 * num_files)
    # files already used in training
    trained = 0

    # ignoring octaves
    note_counts = [0 for i in range(12)]
    total_notes = 0

    # melodies in training / test files
    train_seqs = list()
    test_seqs = list()

    for file in files:
        path = folder + '/' + file
        with open(path, 'r', 0) as f:
            # current melody being processed
            melody = list()
            for line in f:
                words = line.split()
                if words[0] == "Info" and words[1] == "key":
                    if words[2] != "C" or words[3] != "Major": break

                elif words[0] == "Note":
                    # ignores octaves
                    pitch = int(words[3]) % 12
                    melody.append(pitch)
                    if trained < train_max:
                        # if still in training data
                        note_counts[pitch] += 1
                        total_notes += 1

            if melody == []:
                trained += 1
                continue
            elif trained < train_max:
                trained += 1
                train_seqs.append(melody)
            else:
                test_seqs.append(melody)

    probabilities = [0 for i in range(len(note_counts))]

    for note in range(0, len(note_counts)):
        probabilities[note] = note_counts[note] / total_notes

    return (train_seqs, test_seqs, probabilities)

def run_test(sequences, prob_dist):
    P = list()

    for mel in sequences:
        p = 0
        for note in mel:
            p -= log(prob_dist[note])
        p /= len(mel)
        P.append(p)

    mean = reduce(lambda x, y: x + y, P) / len(P)
    P.sort()
    mid = len(P) // 2
    median = P[mid]

    print "Mean: ", mean
    print "Median: ", median

def sum(L):
    total = 0
    for i in L:
        total += i
    return total
            

def main():
    if len(sys.argv) != 2:
        print "Usage: folder containing mel files"
        return 1

    train, test, probs = distributions(sys.argv[1])

    # print "Probability distributions: ", str(probs)
    # print "Sum: ", str(sum(probs))

    print "Testing training data..."
    run_test(train, probs)
    print "Testing test data..."
    run_test(test, probs)

    print "Done."
    return 0

if __name__ == '__main__':
    main()

