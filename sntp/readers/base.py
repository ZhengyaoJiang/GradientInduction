# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import tensorflow as tf
from typing import Optional, List

import logging
logger = logging.getLogger(__name__)


class BaseReader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self,
                 sequence: tf.Tensor,
                 sequence_len: tf.Tensor,
                 is_training: Optional[bool]) -> tf.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_variables(self) -> List[tf.Tensor]:
        raise NotImplementedError
