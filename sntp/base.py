# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
from termcolor import colored

from sntp.util import is_tensor
from sntp.kernels import BaseKernel, RBFKernel
from sntp.store import Store

import logging

from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ProofState:
    def __init__(self,
                 substitutions: Dict[str, tf.Tensor],
                 scores: Optional[tf.Tensor]):
        self._substitutions = substitutions
        self._scores = scores

    @property
    def substitutions(self):
        return self._substitutions

    @property
    def scores(self):
        return self._scores

    @staticmethod
    def _t(t):
        t_min, t_max = tf.reduce_min(t), tf.reduce_max(t)
        content = '{0}, {1:.2f}, {2:.2f}'.format(t.get_shape(), t_min, t_max) if is_tensor(t) else t
        return 'Tensor({})'.format(content)

    @staticmethod
    def _d(d):
        return '{' + ' '.join([key + ': ' + ProofState._t(tensor) for key, tensor in d.items()]) + '}'

    def __str__(self):
        return '{0}: {1}\n{2}: {3}'.format(colored('PS', 'green'), self._t(self.scores),
                                           colored('SUB', 'blue'), self._d(self.substitutions))


class NTPParams:
    def __init__(self,
                 kernel: Optional[BaseKernel] = None,
                 max_depth: int = 1,
                 k_max: Optional[int] = None,
                 mask_indices: Optional[np.ndarray] = None,
                 k_facts: Optional[int] = None,
                 k_rules: Optional[int] = None,
                 index_store: Optional[Store] = None):

        self._kernel = kernel if kernel else RBFKernel()
        self._max_depth = max_depth
        self._k_max = k_max
        self._mask_indices = mask_indices
        self._k_facts = k_facts
        self._k_rules = k_rules
        self._index_store = index_store

    @property
    def kernel(self):
        return self._kernel

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def k_max(self):
        return self._k_max

    @property
    def mask_indices(self):
        return self._mask_indices

    @property
    def k_facts(self):
        return self._k_facts

    @property
    def k_rules(self):
        return self._k_rules

    @property
    def index_store(self):
        return self._index_store
