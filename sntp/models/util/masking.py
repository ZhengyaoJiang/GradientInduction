# -*- coding: utf-8 -*-

import logging

import numpy as np
import tensorflow as tf

from typing import Union

logger = logging.getLogger(__name__)


def create_mask(mask_indices: Union[tf.Tensor, np.ndarray],
                mask_shape: Union[tf.TensorShape, np.ndarray],
                indices: Union[tf.Tensor, np.ndarray]):
    # If there is no indices to mask, return an empty mask
    if tf.equal(tf.reduce_sum(tf.cast(tf.less(mask_indices, 0), tf.int32)), tf.size(mask_indices)):
        return None

    fact_dim = indices.shape[0]
    goal_dim = mask_shape[-1]

    nb_goals = tf.reduce_prod(mask_shape[1:])
    tile_goals = 1 if nb_goals == goal_dim else nb_goals // goal_dim

    mask_shape_2d = [fact_dim, int(nb_goals)]
    fact_indices = tf.reshape(indices, [fact_dim, -1])

    if tile_goals > 1:
        mask_indices = tf.reshape(tf.tile(tf.reshape(mask_indices, [1, -1]), [tile_goals, 1]), [-1])

    condition = tf.greater(mask_indices, -1)

    # Indices of positive examples
    ones_x = tf.boolean_mask(mask_indices, condition)

    # Their position in the batch
    ones_y = tf.cast(tf.squeeze(tf.where(condition)), tf.int32)

    ones_x_tiled = tf.tile(ones_x, [fact_dim])
    ind = tf.reshape(tf.tile(tf.reshape(tf.range(fact_dim), [-1, 1]), [1, ones_x.shape[0]]), [-1])
    ones_y_tiled = tf.tile(ones_y, [fact_dim])

    indices_and_ones_y = tf.transpose(tf.stack([ind, ones_y_tiled]))
    gathered_results = tf.gather_nd(fact_indices, indices_and_ones_y)
    where_equal = tf.where(tf.equal(gathered_results, ones_x_tiled))

    r1 = tf.gather(ind, where_equal)
    r2 = tf.gather(ones_y_tiled, where_equal)

    new_indices = tf.concat([r1, r2], axis=1)

    if new_indices.get_shape()[0] == 0:
        return None

    ones = tf.ones(new_indices.get_shape()[0], dtype=tf.float32)
    mask = tf.scatter_nd(indices=new_indices, updates=ones, shape=mask_shape_2d)
    mask = tf.reshape(mask, mask_shape)

    return 1 - mask
