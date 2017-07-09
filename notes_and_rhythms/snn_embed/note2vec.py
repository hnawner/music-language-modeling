# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import sys
import random

from utils import read_files
from rhythm_utils import read_files_to_embed, build_rhythm_dict
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


def build_dataset(words):
    """Process raw inputs into a dataset."""
    count = collections.Counter(words).most_common() # orders by rarity
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    dictionary[-1] = len(dictionary) # for unkown words
    data = []
    words.append(-1) # for unkown words
    for word in words:
        index = dictionary[word]
        data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def generate_batch(batch_size, num_skips, skip_window, data, data_index):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels, data_index


def note2vec(vocabulary, r_dict, embed_file):
    # vocabulary is list of mels
    # r_dict is rhythm dictionary
    # embed_file is file to save the embedding matrix to
    
    vocabulary = [item for sublist in vocabulary for item in sublist] # flattens
    data, count, dictionary, reversed_dictionary = build_dataset(vocabulary)
    vocabulary_size = len(count) + 1
    del vocabulary  # Hint to reduce memory.

    data_index = 0

    batch_size = 128
    embedding_size = 20   # Dimension of the embedding vector.
    skip_window = 2       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 16    # Number of negative examples to sample.

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

      # Ops and variables pinned to the CPU because of missing GPU implementation
      with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=vocabulary_size))

      # Construct the SGD optimizer using a learning rate of 1.0.
      optimizer = tf.train.GradientDescentOptimizer(.01).minimize(loss)

      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)

      # Add variable initializer.
      init = tf.global_variables_initializer()

    # Step 5: Begin training.
    num_steps = 100001

    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()

      average_loss = 0
      for step in xrange(num_steps):
        batch_inputs, batch_labels, data_index = generate_batch(
            batch_size, num_skips, skip_window, data, data_index)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

      final_embeddings = normalized_embeddings.eval()
      np.savetxt(embed_file, final_embeddings)

    return dictionary
