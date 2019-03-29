# -*- coding: utf-8 -*-

import numpy as np

from typing import List, Optional, Tuple

from sntp.data.util import read_triples, triples_to_vectors
from sntp.data.padding import pad_sequences

from sntp.data import clauses
from sntp.data.clauses import Clause


def parse_clause(text):
    parsed = clauses.grammar.parse(text)
    return clauses.ClauseVisitor().visit(parsed)


class Data:
    def __init__(self,
                 train_path: str,
                 dev_path: Optional[str] = None,
                 test_path: Optional[str] = None,
                 clauses: Optional[List[Clause]] = None,
                 mentions: Optional[List[Tuple[str, str, str]]] = None,
                 test_i_path: Optional[str] = None,
                 test_ii_path: Optional[str] = None):

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        self.test_i_path = test_i_path
        self.test_ii_path = test_ii_path

        self.clauses = clauses if clauses else []
        self.mentions = mentions if mentions else []

        # Loading the dataset
        self.train_triples = read_triples(self.train_path)

        self.dev_triples, self.dev_labels = [], None
        self.test_triples, self.test_labels = [], None

        self.dev_triples = read_triples(self.dev_path) if self.dev_path else []
        self.test_triples = read_triples(self.test_path) if self.test_path else []

        self.test_I_triples = read_triples(self.test_i_path) if self.test_i_path else []
        self.test_II_triples = read_triples(self.test_ii_path) if self.test_ii_path else []

        self.all_triples = self.train_triples + self.dev_triples + self.test_triples

        self.entity_set = {s for (s, _, _) in self.all_triples} | {o for (_, _, o) in self.all_triples}
        self.entity_set |= {s for (s, _, _) in self.mentions} | {o for (_, _, o) in self.mentions}

        self.predicate_set = {p for (_, p, _) in self.all_triples}

        self.pattern_set = {pattern for (_, pattern, _) in self.mentions}
        self.symbol_set = {symbol for pattern in self.pattern_set for symbol in pattern.split(':')}

        self.nb_examples = len(self.train_triples)
        self.nb_mentions = len(self.mentions)

        self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(self.entity_set))}
        self.nb_entities = max(self.entity_to_idx.values()) + 1
        self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}

        self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(self.predicate_set))}
        self.nb_predicates = max(self.predicate_to_idx.values()) + 1
        self.idx_to_predicate = {v: k for k, v in self.predicate_to_idx.items()}

        self.pattern_to_idx = {pattern: idx for idx, pattern in enumerate(sorted(self.pattern_set),
                                                                          start=self.nb_predicates)}
        self.idx_to_pattern = {v: k for k, v in self.pattern_to_idx.items()}

        self.relation_to_idx = {**self.predicate_to_idx, **self.pattern_to_idx}
        self.nb_relations = max(self.relation_to_idx.values()) + 1
        self.idx_to_relation = {v: k for k, v in self.relation_to_idx.items()}

        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(sorted(self.symbol_set))}
        self.nb_symbols = max(self.symbol_to_idx.values()) + 1 if len(self.symbol_set) > 0 else 0
        self.idx_to_symbol = {v: k for k, v in self.symbol_to_idx.items()}

        self.pattern_id_to_symbol_ids = {
            pattern_id: [self.symbol_to_idx[symbol] for symbol in pattern.split(':')]
            for pattern, pattern_id in self.pattern_to_idx.items()
        }

        # Triples
        tri_xs, tri_xp, tri_xo = triples_to_vectors(self.train_triples, self.entity_to_idx, self.predicate_to_idx)
        men_xs, men_xp, men_xo = triples_to_vectors(self.mentions, self.entity_to_idx, self.pattern_to_idx)

        # Triple and mention indices
        self.xi = np.arange(start=0, stop=self.nb_examples + self.nb_mentions, dtype=np.int32)

        self.xs = np.concatenate((tri_xs, men_xs), axis=0)
        self.xp = np.concatenate((tri_xp, men_xp), axis=0)
        self.xo = np.concatenate((tri_xo, men_xo), axis=0)

        assert self.xi.shape == self.xs.shape == self.xp.shape == self.xo.shape

        if len(self.pattern_id_to_symbol_ids) > 0:
            symbol_ids_lst = [s_ids for _, s_ids in sorted(self.pattern_id_to_symbol_ids.items(), key=lambda kv: kv[0])]
            symbol_ids_len_lst = [len(s_ids) for s_ids in symbol_ids_lst]

            self.np_symbol_ids = pad_sequences(symbol_ids_lst)
            self.np_symbol_ids_len = np.array(symbol_ids_len_lst, dtype=np.int32)

        return
