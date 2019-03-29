# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from typing import Optional, Callable

from sntp.data import Data
from sntp.readers import BaseReader

import logging

logger = logging.getLogger(__name__)


class NeuralKB:
    def __init__(self,
                 data: Data,
                 entity_embeddings: tf.Tensor,
                 predicate_embeddings: tf.Tensor,
                 initializer: Optional[Callable] = None,
                 symbol_embeddings: Optional[tf.Tensor] = None,
                 reader: Optional[BaseReader] = None):
        self.relation_embeddings = None
        self.data = data

        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings
        self.symbol_embeddings = symbol_embeddings

        self.initializer = initializer if initializer else tf.random_uniform_initializer(-1.0, 1.0)
        self.predicate_embedding_size = self.predicate_embeddings.get_shape()[-1]

        self.reader = reader

        # before feeding them to a Neural Link Predictor
        self.facts_kb = [data.xp, data.xs, data.xo]
        self.rules_kb = []

        self.variables = []
        name_no = 0

        def variable(shape, name):
            return tfe.Variable(self.initializer(shape), name=name) if shape[0] > 0 else None

        for clause_idx, clause in enumerate(data.clauses):
            self.rule_kb, clause_weight = [], int(clause.weight)

            new_predicate_name_to_var = dict()
            for atom in [clause.head] + list(clause.body):
                predicate_name = atom.predicate.name

                arg1_name = '{}{}'.format(atom.arguments[0].name, clause_idx)
                arg2_name = '{}{}'.format(atom.arguments[1].name, clause_idx)

                if predicate_name in data.predicate_to_idx:
                    predicate_var = [data.predicate_to_idx[predicate_name]] * clause_weight
                else:
                    if predicate_name not in new_predicate_name_to_var:
                        predicate_var = variable(shape=[clause_weight, self.predicate_embedding_size],
                                                 name='predicate_{}'.format(name_no))

                        new_predicate_name_to_var[predicate_name] = predicate_var
                        name_no += 1
                    else:
                        predicate_var = new_predicate_name_to_var[predicate_name]

                self.rule_kb += [[predicate_var, arg1_name, arg2_name]]

            self.variables += [variable for name, variable in new_predicate_name_to_var.items()]
            self.rules_kb += [self.rule_kb]

    def get_variables(self):
        return self.variables

    def create(self):
        self.relation_embeddings = self.predicate_embeddings

        if len(self.data.pattern_id_to_symbol_ids) > 0:
            symbol_seq_embeddings = tf.nn.embedding_lookup(self.symbol_embeddings, self.data.np_symbol_ids)
            pattern_embeddings = self.reader.call(symbol_seq_embeddings, self.data.np_symbol_ids_len)
            self.relation_embeddings = tf.concat([self.predicate_embeddings, pattern_embeddings], axis=0)

        neural_facts_kb = [
            tf.nn.embedding_lookup(self.relation_embeddings, self.facts_kb[0]),
            tf.nn.embedding_lookup(self.entity_embeddings, self.facts_kb[1]),
            tf.nn.embedding_lookup(self.entity_embeddings, self.facts_kb[2]),
        ]

        neural_rules_kb = []
        for rule in self.rules_kb:
            rule_graph = []
            for atom in rule:
                atom_graph = []
                for term in atom:
                    term_is_indices = isinstance(term, list) or isinstance(term, np.ndarray)

                    term_graph = term
                    if term_is_indices is True:
                        term_graph = tf.nn.embedding_lookup(self.relation_embeddings, term)

                    atom_graph += [term_graph]
                rule_graph += [atom_graph]
            neural_rules_kb += [rule_graph]

        return neural_facts_kb, neural_rules_kb
