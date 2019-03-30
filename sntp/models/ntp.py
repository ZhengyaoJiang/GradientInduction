# -*- coding: utf-8 -*-

import tensorflow as tf

from sntp import NTPParams, ProofState

from sntp.models.base import BaseModel
from sntp.kernels import BaseKernel
from sntp.store import Store

from sntp.models.util.prover import neural_or

from typing import List, Union, Optional

import logging
import numpy as np

logger = logging.getLogger(__name__)


class NTP(BaseModel):
    def __init__(self,
                 kernel: BaseKernel,
                 max_depth: int = 1,
                 k_max: Optional[int] = None,
                 facts_k: Optional[int] = None,
                 rules_k: Optional[int] = None,
                 index_store: Optional[Store] = None):
        super().__init__()
        self.kernel = kernel

        self.max_depth = max_depth
        self.k_max = k_max

        self.facts_k = facts_k
        self.rules_k = rules_k

        self.index_store = index_store

        self.initializer = tf.random_uniform_initializer(-1.0, 1.0)
        self.neural_facts_kb = self.neural_rules_kb = None

    def get_variables(self):
        return []

    def predict(self,
                predicate_embeddings: tf.Tensor,
                subject_embeddings: tf.Tensor,
                object_embeddings: tf.Tensor,
                neural_facts_kb: List[List[List[tf.Tensor]]],
                neural_rules_kb: List[List[List[Union[tf.Tensor, str]]]],
                mask_indices: Optional[np.ndarray] = None) -> tf.Tensor:
        """
        Computes the goal scores of input triples (provided as embeddings).

        :param predicate_embeddings: [G, E] tensor of predicate embeddings.
        :param subject_embeddings: [G, E] tensor of subject embeddings.
        :param object_embeddings: [G, E] tensor of object embeddings.
        :param neural_facts_kb: [[K, E], [K, E], [K, E]] tensor list
        :param neural_rules_kb: [[[s_1, s_2, .., s_n]]] list, where each [s_1, s_2, .., s_n] is an atom
        :param mask_indices: [G] vector containing the index of the fact we want to mask in the Neural KB.
        :return: [G] goal scores.
        """
        goals = [predicate_embeddings, subject_embeddings, object_embeddings]
        neural_kb = neural_rules_kb + neural_facts_kb
        start_proof_state = ProofState(substitutions={}, scores=None)

        ntp_params = NTPParams(kernel=self.kernel,
                               max_depth=self.max_depth,
                               k_max=self.k_max,
                               mask_indices=mask_indices,
                               k_facts=self.facts_k,
                               k_rules=self.rules_k,
                               index_store=self.index_store)

        proof_states = neural_or(neural_kb=neural_kb,
                                 goals=goals,
                                 proof_state=start_proof_state,
                                 ntp_params=ntp_params)

        goal_scores_lst = []

        for proof_state in proof_states:
            axis = tf.constant(np.arange(len(proof_state.scores.shape) - 1))
            proof_goal_scores = tf.reduce_max(proof_state.scores, axis=axis)
            goal_scores_lst += [proof_goal_scores]

        maximum_paths = tf.concat([tf.reshape(g, [1, -1]) for g in goal_scores_lst], 0)
        goal_scores = tf.reduce_max(maximum_paths, axis=0)

        return goal_scores
