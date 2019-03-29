# -*- coding: utf-8 -*-

import tensorflow as tf

from typing import Optional

# https://github.com/madalinabuzau/tensorflow-eager-tutorials/blob/master/08_dynamic_recurrent_neural_networks_for_sequence_classification.ipynb


class RNNModel(tf.keras.Model):
    def __init__(self,
                 cell_size: int = 64,
                 dense_size: int = 128,
                 rnn_cell: str = 'lstm',
                 dropout_rate: float = 0.3,
                 out_size: Optional[int] = 100):
        super(RNNModel, self).__init__()
        self.dropout_rate = dropout_rate

        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.zeros_initializer()

        self.dense_layer = tf.keras.layers.Dense(dense_size, activation=tf.nn.relu,
                                                 kernel_initializer=w_init, bias_initializer=b_init)

        self.prediction_layer = None
        if out_size is not None:
            self.prediction_layer = tf.keras.layers.Dense(out_size, activation=None,
                                                          kernel_initializer=w_init,  bias_initializer=b_init)

        cell_name_to_class = {
            'lstm': tf.nn.rnn_cell.LSTMCell,
            'ugrnn': tf.contrib.rnn.UGRNNCell,
            'gru': tf.nn.rnn_cell.GRUCell
        }

        assert rnn_cell in cell_name_to_class
        rnn_cell_class = cell_name_to_class[rnn_cell]
        self.rnn_cell = rnn_cell_class(cell_size)
        return

    def predict_(self,
                 sequence: tf.Tensor,
                 sequence_length: tf.Tensor,
                 is_training: bool) -> tf.Tensor:
        nb_samples = tf.shape(sequence)[0]
        state = self.rnn_cell.zero_state(nb_samples, dtype=sequence.dtype)
        unstacked_embeddings = tf.unstack(sequence, axis=1)
        outputs = []

        for input_step in unstacked_embeddings:
            output, state = self.rnn_cell(input_step, state)
            outputs.append(output)

        outputs = tf.stack(outputs, axis=1)
        idxs_last_output = tf.stack([tf.range(nb_samples), tf.cast(sequence_length - 1, tf.int32)], axis=1)
        final_output = tf.gather_nd(outputs, idxs_last_output)

        dropped_output = final_output
        if self.dropout_rate is not None:
            dropped_output = tf.layers.dropout(final_output, rate=self.dropout_rate, training=is_training)

        dense = self.dense_layer(dropped_output)

        logits = dense
        if self.prediction_layer is not None:
            logits = self.prediction_layer(dense)

        return logits
