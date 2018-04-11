from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from core.rules import RulesManager
from core.clause import Predicate

tf.enable_eager_execution()

class Agent(object):
    def __init__(self, rules_manager, background):
        self.rules_manager = rules_manager
        self.labels = None
        self.rule_weights = {} # dictionary from predicates to rule weights matrices
        self.__init__rule_weights()
        self.ground_atoms = rules_manager.all_grounds
        self.base_valuation = self.axioms2valuation(background)

    def __init__rule_weights(self):
        with tf.variable_scope("rule_weights", reuse=tf.AUTO_REUSE):
            for predicate, clauses in self.rules_manager.all_clauses.items():
                self.rule_weights[predicate] = tf.get_variable(predicate.name+"_rule_weights",
                                                               [len(clauses[0]), len(clauses[1])],
                                                               dtype=tf.float32)

    def axioms2valuation(self, axioms):
        '''
        :param axioms: list of Atoms, background knowledge
        :return: a valuation vector
        '''
        result = np.zeros(len(self.ground_atoms), dtype=np.float32)
        for i, atom in enumerate(self.ground_atoms):
            if atom in axioms:
                result[i] = 1.0
        return result

    def deduction(self):
        # takes background as input and return a valuation of target ground atoms
        pass

    def inference_step(self, valuation):
        deduced_valuation = np.zeros(len(self.ground_atoms))
        # deduction_matrices = self.rules_manager.deducation_matrices[predicate]
        for predicate, matrix in self.rules_manager.deduction_matrices.items():
            deduced_valuation += Agent.inference_single_predicate(valuation, matrix, self.rule_weights[predicate])
        return deduced_valuation+valuation - deduced_valuation*valuation


    @staticmethod
    def inference_single_predicate(valuation, deduction_matrices, rule_weights):
        '''

        :param valuation:
        :param deduction_matrices: list of list of matrices
        :param rule_weights: tensor, shape (number_of_first_clauses, number_of_second_clauses)
        :return:
        '''
        result_valuations = [[], []]
        for i in range(len(result_valuations)):
            for matrix in deduction_matrices[i]:
                result_valuations[i].append(Agent.inference_single_clause(valuation, matrix))

        c_p = [] # flattened
        for clause1 in result_valuations[0]:
            for clause2 in result_valuations[1]:
                c_p.append(tf.maximum(clause1, clause2))
        rule_weights = tf.reshape(rule_weights ,[-1, 1])
        return tf.reduce_mean((tf.stack(c_p)*tf.nn.softmax(rule_weights)), axis=0)

    @staticmethod
    def inference_single_clause(valuation, X):
        '''
        The F_c in the paper
        :param valuation:
        :param X: array, size (number)
        :return: tensor, size (number_of_ground_atoms)
        '''
        X1 = X[:, :, 0, None]
        X2 = X[:, :, 1, None]
        Y1 = tf.gather_nd(params=valuation, indices=X1)
        Y2 = tf.gather_nd(params=valuation, indices=X2)
        Z = Y1*Y2
        return tf.reduce_max(Z, axis=1)

    def loss(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(self.deduction(), self.labels)


if __name__ == "__main__":
    from core.clause import *
    from core.ilp import *
    from core.rules import *

    constants = [str(i) for i in range(10)]
    background = [Atom(Predicate("succ", 2), [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(Predicate("zero", 1), "0"))
    positive = [Atom(Predicate("predecessor", 2), [constants[i + 1], constants[i]]) for i in range(9)]
    all_atom = [Atom(Predicate("predecessor", 2), [constants[i], constants[j]]) for i in range(9) for j in range(9)]
    negative = list(set(all_atom) - set(positive))

    language = LanguageFrame(Predicate("predecessor",2), [Predicate("zero",1), Predicate("succ",2)], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([], {Predicate("predecessor", 2): [RuleTemplate(1, True), RuleTemplate(1, True)]},
                                   10)
    man = RulesManager(language, program_temp)
    atoms = man.generate_body_atoms(Predicate("predecessor", 2), ("X", "Y"), ("X"))

    clauses = man.generate_clauses(Predicate("predecessor", 2), RuleTemplate(1, True))

    agent = Agent(man, background)

    print(agent.inference_step(agent.base_valuation))