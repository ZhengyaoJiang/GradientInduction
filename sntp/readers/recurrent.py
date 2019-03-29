# -*- coding: utf-8 -*-

import tensorflow as tf

from sntp.readers.base import BaseReader
from sntp.readers.models import RNNModel

from typing import Optional


class RecurrentReader(BaseReader):
    def __init__(self,
                 cell_size: int = 64,
                 dense_size: int = 128,
                 out_size: int = 100,
                 rnn_cell: str = 'lstm',
                 dropout_rate: float = 0.3):
        super(RecurrentReader, self).__init__()
        self.model = RNNModel(cell_size=cell_size,
                              dense_size=dense_size,
                              out_size=out_size,
                              rnn_cell=rnn_cell,
                              dropout_rate=dropout_rate)
        return

    def __call__(self,
                 sequence: tf.Tensor,
                 sequence_len: tf.Tensor,
                 is_training: Optional[bool]) -> tf.Tensor:
        return self.model.predict_(sequence, sequence_len, is_training=is_training)

    def get_variables(self):
        return self.model.variables
