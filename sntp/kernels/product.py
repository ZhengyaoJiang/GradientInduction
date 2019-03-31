# -*- coding: utf-8 -*-

import tensorflow as tf
from sntp.kernels.base import BaseKernel

import logging

logger = logging.getLogger(__name__)


class ProductKernel(BaseKernel):
    def __init__(self):
        super().__init__()

    def pairwise(self, x, y):
        dim_x, emb_size_x = x.get_shape()[:-1], x.get_shape()[-1]
        dim_y, emb_size_y = y.get_shape()[:-1], y.get_shape()[-1]

        a = tf.reshape(x, [-1, emb_size_x])
        b = tf.reshape(y, [-1, emb_size_y])
        #entity_diff = tf.einsum('x,y->xy', a[:, -1], (b[:, -1]))-a[:,-1,None]*a[:,-1,None]
        dot = tf.einsum('xe,ye->xy', a, b)
        #similarity = tf.cast(tf.abs(entity_diff)<1e-2, dtype=tf.float32) if tf.reduce_sum(dot)<1e-8 else dot

        return tf.reshape(dot, dim_x.concatenate(dim_y))

    def __call__(self, x, y):
        emb_size_x = x.get_shape()[-1]
        emb_size_y = y.get_shape()[-1]

        a = tf.reshape(x, [-1, emb_size_x])
        b = tf.reshape(y, [-1, emb_size_y])

        dot = tf.einsum('xe,xe->x', a, b)
        #similarity = tf.cast(tf.equal(a[:,-1], b[:,-1]), dtype=tf.float32) if tf.reduce_sum(dot)<1e-8 else dot
        return dot