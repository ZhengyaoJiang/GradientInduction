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
    def __init__(self, clauses, embeddings):
        """
        :param clauses: all clauses, including facts! facts are represented as a
        clause with empty body.
        """
        self.__embeddings = embeddings
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
        substitution = state.substitution[:]
        score = state.score
        for i in range(atom1.arity):
            term1 = atom1.terms[i]
            term2 = atom2.terms[i]
            if is_variable(term1) and is_variable(term2):
                pass
            elif is_variable(term1):
                substitution.append((term1, term2))
            elif is_variable(term2):
                substitution.append((term2, term1))
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
        if not isinstance(state, ProofState):
            raise ValueError()
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
        if not isinstance(state, ProofState):
            raise ValueError()
        if tuple(state)==tuple(FAIL):
            return [FAIL]
        if depth==0:
            return [FAIL]
        if len(body)==0:
            return [state]
        states = []
        for i, atom in enumerate(body):
            or_states = self.apply_rules(NeuralProver.substitute(atom, state.substitution),
                                      depth-1, state)
            for or_state in or_states:
                states.extend(self.apply_rule(body[i+1:],depth,or_state))
        return states

class VariableManager():
    def __init__(self):
        self.__max_id = 0

    def activate(self, clause):
        activated_clause = clause.assign_var_id(self.__max_id)
        self.__max_id += len(clause.variables)
        return activated_clause

class Embeddings():
    def __init__(self, predicates, para_predicates, constants, dimension=5):
        self.predicates = set(predicates)
        self.constants = set(constants)
        self.para_predicates = set(para_predicates)
        self.embbedings = {}
        for predicate in predicates:
            self.embbedings[predicate] = tf.get_variable(predicate.name,shape=[dimension],dtype=tf.float32)
        for constant in constants:
            self.embbedings[constant] = tf.get_variable(constant,shape=[dimension],dtype=tf.float32)

    def __getitem__(self, key):
        return self.embbedings[key]

    @staticmethod
    def from_clauses(clauses, para_clauses):
        predicates = set()
        constants = set()
        para_predicates = set()
        for clause in clauses:
            predicates.update(clause.predicates)
            constants.update(clause.constants)
        for para_clause in para_clauses:
            const = para_clause.constants
            if not const.issubset(constants):
                raise ValueError("parameterized clause shouldn't include the constants that didn't appear"
                                 "in main clauses")
            para_predicates.update(para_clause.predicates)
        return Embeddings(predicates, para_predicates, constants)

if __name__ == "__main__":
    from core.clause import str2clause,str2atom
    clause_str = ["fatherOf(abe, homer)","parentOf(homer,cart)",
                  "grandFatherOf(X,Y):-fatherOf(X,Z),parentOf(Z,Y)"]
    clauses = [str2clause(s) for s in clause_str]
    # para_clauses = [str2clause("r(X,Y):-p(X,Z),q(Z,Y)")]
    para_clauses = []
    embeddings = Embeddings.from_clauses(clauses, para_clauses)
    ntp = NeuralProver(clauses, embeddings)
    states = ntp.apply_rules(str2atom("grandFatherOf(abe,cart)"),2,ProofState([],1))
    print(states)