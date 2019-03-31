# -*- coding: utf-8 -*-

import copy
import numpy as np

import tensorflow as tf

from sntp import ProofState
from sntp.util import is_tensor, is_variable

from typing import List, Union


def k_max(goal: List[Union[tf.Tensor, str]],
          proof_state: ProofState,
          k: int = 10):
    new_proof_state = proof_state

    goal_variables = {goal_elem for goal_elem in goal if is_variable(goal_elem)}

    if len(goal_variables) > 0:
        # Check goal for variables
        scores = proof_state.scores
        substitutions = copy.copy(proof_state.substitutions)

        scores_shp = scores.get_shape()  # [K, R, G]
        k_size = scores_shp[0]  # K

        # R * G
        batch_size = tf.reduce_prod(scores_shp[1:])

        # [ k, R, G ]
        new_scores_shp = tf.TensorShape(k).concatenate(scores_shp[1:])

        # [K, R, G] -> [K, R * G]
        scores_2d = tf.reshape(scores, [k_size, -1])

        # [R * G, K]
        scores_2d_t = tf.transpose(scores_2d)
        # [ R * G, k]

        scores_2d_top_k_t, scores_2d_top_k_idxs_t = tf.nn.top_k(scores_2d_t, k=k)

        # [ k, R * G ]
        scores_2d_top_k = tf.transpose(scores_2d_top_k_t)
        # [ k, R * G ]
        scores_2d_top_k_idxs = tf.transpose(scores_2d_top_k_idxs_t)

        # [k, R, G]
        scores_top_k = tf.reshape(scores_2d_top_k, new_scores_shp)

        # [[ 4 23 10  1 29], [14 30 14 20 21]]
        coordinates_lhs = scores_2d_top_k_idxs
        # [[ 4 14] [23 30] [10 14] [ 1 20] [29 21]],
        coordinates_lhs = tf.reshape(coordinates_lhs, [1, -1])
        # [[ 4 14 23 30 10 14  1 20 29 21]]

        # [0, 1, 2, 3, 4]
        coordinates_rhs = tf.reshape(tf.range(0, batch_size), [1, -1])
        # [[0], [1], [2], [3], [4]]
        coordinates_rhs = tf.tile(coordinates_rhs, [1, k])
        # [[0 0] [1 1] [2 2] [3 3] [4 4]]

        k_2d_coordinates = tf.concat([coordinates_lhs, coordinates_rhs], axis=0)
        # [[ 4 14 23 30 10 14  1 20 29 21]
        #  [ 0  0  1  1  2  2  3  3  4  4]]
        k_2d_coordinates = tf.transpose(k_2d_coordinates)

        # Assume we unified [KE, KE, KE] and [GE, GE, X] - we get X/KGE.
        # Reshape X such that we have X/K'GE instead.
        for goal_variable in {g for g in goal_variables if g in substitutions}:
            var_tensor = substitutions[goal_variable]

            if is_tensor(var_tensor):
                # Variable is going to be [K, R, G, E]
                substitution_shp = var_tensor.get_shape()
                embedding_size = substitution_shp[-1]
                #TODO: problem with k-max
                assert substitution_shp[0] == k_size

                new_substitution_shp = new_scores_shp.concatenate([embedding_size])

                # Reshape to [K, R * G, E]
                substitution_3d = tf.reshape(var_tensor, [k_size, -1, embedding_size])

                new_substitution_3d = tf.gather_nd(substitution_3d, k_2d_coordinates)
                new_substitution_3d = tf.reshape(new_substitution_3d, [k, -1, embedding_size])

                substitutions[goal_variable] = tf.reshape(new_substitution_3d, new_substitution_shp)

        new_proof_state = ProofState(scores=scores_top_k,
                                     substitutions=substitutions)
    return new_proof_state
