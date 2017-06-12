#!/usr/bin/env python2.6
#SBATCH --partition=standard

from __future__ import division, print_function
import os, sys
from math import log
import offsets

def distributions(folder):
    # list of files in input directory
    files = os.listdir(folder)
    num_files = len(files)

    # number of files used in training set
    train_max = int(0.7 * num_files)
    print("Partition: ", train_max)
    # files already used in training
    trained = 0

    train_seqs = list()
    test_seqs = list()

    # temp variable for tracking major-key melodies
    num_melodies_recorded = 0

    # ignoring octaves
    matrix = [ [0] * 12 for i in range(12) ]
    note_counts = [0] * 12
    start_counts = [0] * 12

    for file in os.listdir(folder):
        path = folder + "/" + file
        offset = 0 # offset from key of C
        with open(path, 'r', 0) as f:
            melody = list()
            prev_note = -1
            for line in f:
                words = line.split()

                if words[0] == "Info" and words[1] == "key":
                    if words[3] == "Major":
                        offset = offsets.d[words[2]]
                    else: break

                elif words[0] == "Note":
                    # ignores octaves
                    pitch = (int(words[3]) - offset) % 12
                    melody.append(pitch)
                    if trained < train_max:
                        # if still in training data
                        if prev_note == -1:
                            start_counts[pitch] += 1
                        else:
                            note_counts[pitch] += 1
                            matrix[prev_note][pitch] += 1
                        prev_note = pitch

            if melody == []:
                trained += 1
            elif trained < train_max:
                trained += 1
                num_melodies_recorded += 1
                train_seqs.append(melody)
            else:
                test_seqs.append(melody)

    print("Training sequences: ", len(train_seqs))
    print("Testing sequences: ", len(test_seqs))

    #-- Probability calculations --#

    p_matrix = [ ([0] * 12) for i in range(len(note_counts)) ]
    p_start = [ 0 for i in range(len(start_counts)) ]

    for note1 in range(0, len(note_counts)):
        p_start[note1] = start_counts[note1] / num_melodies_recorded
        for note2 in range(0, len(note_counts)):
            p_matrix[note1][note2] = matrix[note1][note2] / note_counts[note1]

    return (train_seqs, test_seqs, p_matrix, p_start)

def genre_match(sequences, p_matrix, p_start):
    P = list()

    for mel in sequences:
        p = 0
        prev = -1
        for note in mel:
            if prev == -1:
                if p_start[note] != 0:
                    p -= log(p_start[note])
            else:
                if p_matrix[prev][note] != 0:
                    p -= log(p_matrix[prev][note])
            prev = note
        p /= len(mel)
        P.append(p)

    mean = reduce(lambda x, y: x + y, P) / len(P)
    P.sort()
    mid = len(P) // 2
    median = P[mid]

    print("Mean: ", mean)
    print("Median: ", median, "\n")

def predict(sequences, p_matrix):
    correct = 0
    predictions = 0

    for mel in sequences:
        for index in range(1, len(mel)):
            prev = mel[index-1]
            note = 0
            for i in range(1, len(p_matrix[prev])):
                if p_matrix[prev][note] < p_matrix[prev][i]:
                    note = i

            if mel[index] == note:
                correct += 1

            predictions += 1

    print("Total predictions: ", predictions)
    print("Total correct predictions: ", correct)
    print("Percentage correct: ", correct / predictions, "\n")

#--- debug ---#
def print_list(L):
    print("[", end="")
    for item in L:
        print(item, end=" ")
    print("]")


def main():
    if len(sys.argv) != 2:
        print("Usage: folder containing mel files")
        return 1

    train, test, p_matrix, p_start = distributions(sys.argv[1])

    print("Testing training data: genre matching...")
    genre_match(train, p_matrix, p_start)
    print("Testing test data: genre matching...")
    genre_match(test, p_matrix, p_start)
    print("Testing training data: predictions...")
    predict(train, p_matrix)
    print("Testing test data: predictions...")
    predict(test, p_matrix)

    print("Done.")
    return 0

if __name__ == '__main__':
    main()

