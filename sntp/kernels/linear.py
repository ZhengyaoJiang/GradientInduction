# -*- coding: utf-8 -*-

import tensorflow as tf

from sntp.kernels.base import BaseKernel

import logging

logger = logging.getLogger(__name__)


class LinearKernel(BaseKernel):
    def __init__(self):
        super().__init__()

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

        dot = tf.einsum('xe,xe->x', a, b)
        return tf.nn.sigmoid(dot)
