from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf
from collections import OrderedDict
import tensorflow.contrib.eager as tfe
from core.clause import is_variable

tf.enable_eager_execution()

from collections import namedtuple

ProofState = namedtuple("ProofState", "substitution score")
"""
substitution is a list of binary tuples, where the first element is the
 variable and second one is a constant.
score is a float (Tensor) representing the sucessness of the proof.
"""
FAIL = ProofState(set(), -1)

class NeuralProver():
    def __init__(self, clauses):
        """
        :param clauses: all clauses, including facts! facts are represented as a
        clause with empty body.
        """
        self.__embeddings = {}
        self.__clauses = clauses
        self.__var_manager = VariableManager()

    def unify(self, atom1, atom2, state):
        """
        :param atom1: 
        :param atom2: 
        :param state: 
        :return: result proof state with substituted variables and new scores 
        """
        if atom1.arity != atom2.arity:
            return FAIL
        substitution = state.substitution.copy()
        score = state.score
        for i in range(atom1.arity):
            term1 = atom1.terms[i]
            term2 = atom2.terms[i]
            if is_variable(term1) and is_variable(term2):
                pass
            elif is_variable(term1):
                substitution = substitution.union((term1, term2))
            elif is_variable(term2):
                substitution = substitution.union((term2, term1))
            else:
                score = tf.minimum(score,
                                   tf.exp(-tf.reduce_sum(
                                       (self.__embeddings[term1]- self.__embeddings[term2])**2)))
        return ProofState(substitution, score)

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
            clause = self.__var_manager.activate(clause)
            states.extend(self.apply_rule(
                clause.body, depth, self.unify(clause.head, goal, state)))
        return states

    @staticmethod
    def substitute(atom, substitution):
        """
        substitute variables in an atom given the list of substitution pairs
        :param atom:
        :param substitution: list of binary tuples
        :return:
        """
        replace_dict = {pair[0]: pair[1] for pair in substitution}
        return atom.replace(replace_dict)

    def apply_rule(self, body, depth, state):
        """
        the original and module.
        Loop through all atoms of the body and apply apply_rules on each atom.
        :param body: the list of subgoals
        :param depth:
        :param state:
        :return:
        """
        if state==FAIL:
            return FAIL
        if depth==0:
            return FAIL
        if len(body)==0:
            return state
        states = []
        for i, atom in enumerate(body):
            or_states = self.apply_rules(NeuralProver.substitute(atom, state.substitution),
                                      depth-1, state)
            for or_state in or_states:
                states.extend(self.apply_rule(body[i+1:],depth,or_state))
        return states

class VariableManager():
    def __init__(self):
        self.max_id = 0

    def activate(self, clause):
        self.max_id += len(clause.variables)
        return clause.assign_var_id(self.max_id)


