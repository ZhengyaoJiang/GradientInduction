# -*- coding: utf-8 -*-

import tensorflow as tf

from sntp.readers.base import BaseReader
from sntp.readers.models import AverageModel

from typing import Optional


class AverageReader(BaseReader):
    def __init__(self,
                 out_size: int = 100):
        super(AverageReader, self).__init__()
        self.model = AverageModel(out_size=out_size)
        return

    def __call__(self,
                 sequence: tf.Tensor,
                 sequence_len: tf.Tensor,
                 is_training: Optional[bool]) -> tf.Tensor:
        return self.model.predict_(sequence, sequence_len, is_training=is_training)

    def get_variables(self):
        return self.model.variables
