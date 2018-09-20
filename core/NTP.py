from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf
from collections import OrderedDict
import tensorflow.contrib.eager as tfe
from core.clause import is_variable

tf.enable_eager_execution()

from collections import namedtuple

ProofState = namedtuple("ProofState", "substitution score")
FAIL = ProofState(set(), -1)

class NeuralProver():
    def __init__(self, clauses):
        self.__embeddings = {}
        self.__clauses = clauses

    def unify(self, atom1, atom2, state):
        """
        :param atom1: 
        :param atom2: 
        :param state: 
        :return: result proof state with substituted variables and new scores 
        """
        if atom1.arity != atom2.arity:
            return FAIL
        for i in range(atom1.arity):
            term1 = atom1.terms[i]
            term2 = atom2.terms[i]
            if is_variable(term1) and is_variable(term2):
                return state
            elif is_variable(term1):
                return ProofState(state.substitution.union((term1, term2)),
                                  state.score)
            elif is_variable(term2):
                return ProofState(state.substitution.union((term2, term1)),
                                  state.score)
            else:
                return ProofState(state.substitution, tf.minimum(state.score,
                                  tf.exp(-tf.reduce_sum((self.__embeddings[term1]-
                                                         self.__embeddings[term2])**2))))

    def apply_rules(self, goal, depth, state):
        """
        the or module in the original article
        :param goal: 
        :param depth: 
        :param state: 
        :return: list of states
        """
        states = []
        for clause in self.__clauses:
            states.extend(self.apply_rule(
                clause.body, depth, self.unify(clause.head, goal, state)))
        return states

    @staticmethod
    def substitute(atom, substitution):
        replace_dict = {pair[0]: pair[1] for pair in substitution}


    def apply_rule(self, body, depth, state):
        states = []

        return states




