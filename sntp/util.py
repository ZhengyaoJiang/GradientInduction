# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from termcolor import colored
from collections.abc import Iterable


def is_tensor(atom_elem):
    return isinstance(atom_elem, tf.Tensor) or isinstance(atom_elem, tf.Variable)


def is_variable(atom_elem):
    return isinstance(atom_elem, str) and atom_elem.isupper()


def atom_to_str(atom):
    if isinstance(atom, Iterable):
        def _to_show(e):
            return e.get_shape() if is_tensor(e) else e.shape if isinstance(e, np.ndarray) else e
        body_str = str([_to_show(e) for e in atom])
        body_str = body_str.replace('Dimension', '')
    else:
        body_str = str(atom)
    return '{} {}'.format(colored('A', 'red'), body_str)


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res


def corrupt_triples(random_state: np.random.RandomState,
                    xs_batch: np.ndarray,
                    xp_batch: np.ndarray,
                    xo_batch: np.ndarray,
                    xs: np.ndarray,
                    xp: np.ndarray,
                    xo: np.ndarray,
                    entity_indices: np.ndarray,
                    corrupt_subject: bool = False,
                    corrupt_object: bool = False):
    true_triples = {(s, p, o) for s, p, o in zip(xs, xp, xo)}

    res_xs, res_xp, res_xo = [], [], []
    for s, p, o in zip(xs_batch, xp_batch, xo_batch):
        corrupt_s = s
        corrupt_p = p
        corrupt_o = o

        done = False
        while not done:
            if corrupt_subject is True:
                corrupt_s = entity_indices[random_state.choice(entity_indices.shape[0])]

            if corrupt_object is True:
                corrupt_o = entity_indices[random_state.choice(entity_indices.shape[0])]

            done = (corrupt_s, corrupt_p, corrupt_o) not in true_triples

        res_xs += [corrupt_s]
        res_xp += [corrupt_p]
        res_xo += [corrupt_o]

    return np.array(res_xs), np.array(res_xp), np.array(res_xo)
