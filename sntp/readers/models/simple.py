# -*- coding: utf-8 -*-

import tensorflow as tf

from typing import Optional

from sntp.readers.models.tfutil import mean


class AverageModel(tf.keras.Model):
    def __init__(self,
                 out_size: Optional[int] = 100):
        super(AverageModel, self).__init__()
        self.out_size = out_size

        self.prediction_layer = None
        if self.out_size is not None:
            w_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.zeros_initializer()
            self.prediction_layer = tf.keras.layers.Dense(out_size, activation=None,
                                                          kernel_initializer=w_init, bias_initializer=b_init)
        return

    def predict_(self,
                 sequence: tf.Tensor,
                 sequence_length: tf.Tensor,
                 is_training: bool) -> tf.Tensor:
        average = mean(sequence, sequence_length)
        logits = self.prediction_layer(average) if self.out_size is not None else average
        return logits
