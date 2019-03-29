# -*- coding: utf-8 -*-

import numpy as np

from sntp.data import Data
from sntp.util import make_batches, corrupt_triples


class Batcher:
    def __init__(self,
                 data: Data,
                 batch_size: int,
                 nb_epochs: int,
                 random_state: np.random.RandomState,
                 nb_corrupted_pairs: int = 2,
                 is_all: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.nb_corrupted_pairs = nb_corrupted_pairs
        self.is_all = is_all
        self.random_state = random_state

        size = nb_epochs * data.nb_examples
        self.curriculum_xi = np.zeros(size, dtype=np.int32)
        self.curriculum_xs = np.zeros(size, dtype=np.int32)
        self.curriculum_xp = np.zeros(size, dtype=np.int32)
        self.curriculum_xo = np.zeros(size, dtype=np.int32)

        for epoch_no in range(nb_epochs):
            curriculum_order = self.random_state.permutation(data.nb_examples)
            start = epoch_no * data.nb_examples
            end = (epoch_no + 1) * data.nb_examples
            self.curriculum_xi[start: end] = data.xi[curriculum_order]
            self.curriculum_xs[start: end] = data.xs[curriculum_order]
            self.curriculum_xp[start: end] = data.xp[curriculum_order]
            self.curriculum_xo[start: end] = data.xo[curriculum_order]

        self.batches = make_batches(self.curriculum_xi.shape[0], batch_size)
        self.nb_batches = len(self.batches)

        self.entity_indices = np.array(sorted({data.entity_to_idx[entity] for entity in data.entity_set}))

    def get_batch(self, batch_start, batch_end):
        current_batch_size = batch_end - batch_start

        # Let's keep the batches like this:
        # Positive, Negative, Negative, Positive, Negative, Negative, ..
        nb_negatives = self.nb_corrupted_pairs * 2 * (2 if self.is_all else 1)
        nb_triple_variants = 1 + nb_negatives

        xi_batch = np.full(shape=current_batch_size * nb_triple_variants, fill_value=-1, dtype=self.curriculum_xi.dtype)
        xs_batch = np.zeros(current_batch_size * nb_triple_variants, dtype=self.curriculum_xs.dtype)
        xp_batch = np.zeros(current_batch_size * nb_triple_variants, dtype=self.curriculum_xp.dtype)
        xo_batch = np.zeros(current_batch_size * nb_triple_variants, dtype=self.curriculum_xo.dtype)

        # Indexes of positive examples in the Neural KB
        xi_batch[0::nb_triple_variants] = self.curriculum_xi[batch_start:batch_end]

        # Positive examples
        xs_batch[0::nb_triple_variants] = self.curriculum_xs[batch_start:batch_end]
        xp_batch[0::nb_triple_variants] = self.curriculum_xp[batch_start:batch_end]
        xo_batch[0::nb_triple_variants] = self.curriculum_xo[batch_start:batch_end]

        def corrupt(**kwargs):
            res = corrupt_triples(self.random_state,
                                  self.curriculum_xs[batch_start:batch_end],
                                  self.curriculum_xp[batch_start:batch_end],
                                  self.curriculum_xo[batch_start:batch_end],
                                  self.data.xs, self.data.xp, self.data.xo,
                                  self.entity_indices, **kwargs)
            return res

        for c_index in range(0, self.nb_corrupted_pairs):
            c_i = c_index * 2 * (2 if self.is_all else 1) + 1

            # Let's corrupt the subject of the triples
            xs_corr, xp_corr, xo_corr = corrupt(corrupt_subject=True)
            xs_batch[c_i::nb_triple_variants] = xs_corr
            xp_batch[c_i::nb_triple_variants] = xp_corr
            xo_batch[c_i::nb_triple_variants] = xo_corr

            # Let's corrupt the object of the triples
            xs_corr, xp_corr, xo_corr = corrupt(corrupt_object=True)
            xs_batch[c_i + 1::nb_triple_variants] = xs_corr
            xp_batch[c_i + 1::nb_triple_variants] = xp_corr
            xo_batch[c_i + 1::nb_triple_variants] = xo_corr

            if self.is_all:
                # Let's corrupt the subject of the triples
                xs_corr, xp_corr, xo_corr = corrupt(corrupt_subject=True, corrupt_object=True)
                xs_batch[c_i + 2::nb_triple_variants] = xs_corr
                xp_batch[c_i + 2::nb_triple_variants] = xp_corr
                xo_batch[c_i + 2::nb_triple_variants] = xo_corr

                # Let's corrupt the object of the triples
                xs_corr, xp_corr, xo_corr = corrupt(corrupt_subject=True, corrupt_object=True)
                xs_batch[c_i + 3::nb_triple_variants] = xs_corr
                xp_batch[c_i + 3::nb_triple_variants] = xp_corr
                xo_batch[c_i + 3::nb_triple_variants] = xo_corr

        targets = np.array(([1] + ([0] * nb_negatives)) * current_batch_size, dtype='float32')

        return xi_batch, xp_batch, xs_batch, xo_batch, targets
