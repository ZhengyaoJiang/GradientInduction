# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def mean(sequence: tf.Tensor,
         sequence_length: tf.Tensor) -> tf.Tensor:
    sequence_shape = sequence.get_shape()
    batch_size, max_len, embedding_size = sequence_shape[0], sequence_shape[1], sequence_shape[2]

    mask = tf.sequence_mask(sequence_length, maxlen=max_len)
    mask = tf.reshape(mask, [batch_size, max_len, 1])
    mask = tf.tile(mask, [1, 1, embedding_size])
    mask = tf.cast(mask, dtype=sequence.dtype)

    masked_seq_emb = sequence * mask

    average = tf.reduce_sum(masked_seq_emb, axis=1)
    average = average / tf.cast(tf.expand_dims(sequence_length, axis=1), dtype=sequence.dtype)
    return average


def max1d(sequence: tf.Tensor,
          sequence_length: tf.Tensor) -> tf.Tensor:
    seq_shape = sequence.get_shape()
    batch_size, max_len, embedding_size = seq_shape[0], seq_shape[1], seq_shape[2]

    mask = tf.sequence_mask(sequence_length, maxlen=max_len)
    mask = tf.reshape(mask, [batch_size, max_len, 1])
    mask = tf.tile(mask, [1, 1, embedding_size])
    mask = tf.cast(mask * (- np.inf), dtype=sequence.dtype)

    return tf.reduce_max(sequence * mask, axis=1)


def reverse_sequence(sequence: tf.Tensor,
                     sequence_length: tf.Tensor) -> tf.Tensor:
    return tf.reverse_sequence(sequence, sequence_length, batch_axis=0, seq_axis=1)
