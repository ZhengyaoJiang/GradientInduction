# -*- coding: utf-8 -*-

import tensorflow as tf

from sntp.kernels.base import BaseKernel

import logging

logger = logging.getLogger(__name__)


class RBFKernel(BaseKernel):
    def __init__(self, slope=1.0):
        super().__init__()
        self.slope = slope

    def __call__(self, x, y):
        """
        :param x: [AxE] Tensor
        :param y: [AxE] Tensor
        :return: [A] Tensor
        """
        emb_size_x = x.get_shape()[-1]
        emb_size_y = y.get_shape()[-1]

        a = tf.reshape(x, [-1, emb_size_x])
        b = tf.reshape(y, [-1, emb_size_y])

        l2 = tf.reduce_sum(tf.square(a - b), 1)

        l2 = tf.clip_by_value(l2, 1e-6, 1000)
        l2 = tf.sqrt(l2)
        return tf.exp(- l2 * self.slope)
