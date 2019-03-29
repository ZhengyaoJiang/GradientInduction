# -*- coding: utf-8 -*-

import gzip
import bz2

import numpy as np

import logging

logger = logging.getLogger(__name__)


def iopen(file, *args, **kwargs):
    f = open
    if file.endswith('.gz'):
        f = gzip.open
    elif file.endswith('.bz2'):
        f = bz2.open
    return f(file, *args, **kwargs)


def read_triples(path):
    triples = []
    with iopen(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split()
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples


def triples_to_vectors(triples, entity_to_idx, predicate_to_idx):
    xs = np.array([entity_to_idx[s] for (s, p, o) in triples], dtype=np.int32)
    xp = np.array([predicate_to_idx[p] for (s, p, o) in triples], dtype=np.int32)
    xo = np.array([entity_to_idx[o] for (s, p, o) in triples], dtype=np.int32)
    return xs, xp, xo
