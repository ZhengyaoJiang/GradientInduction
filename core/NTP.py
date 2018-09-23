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
substitution is a set of binary tuples, where the first element is the
 variable and second one is a constant.
score is a float (Tensor) representing the sucessness of the proof.
"""
FAIL = ProofState(set(), 0)

class NeuralProver():
    def __init__(self, clauses, embeddings):
        """
        :param clauses: all clauses, including facts! facts are represented as a
        clause with empty body.
        """
        self.__embeddings = embeddings
        self.__clauses = clauses
        self.__var_manager = VariableManager()

    def prove(self, goal, depth):
        initial_state = ProofState(set(), 1)
        states = self.apply_rules(goal, depth, initial_state)
        scores = tf.stack([state.score for state in states])
        return tf.reduce_max(scores)

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
        for i in range(atom1.arity+1):
            if i==0:
                symbol1 = atom1.predicate
                symbol2 = atom2.predicate
            else:
                symbol1 = atom1.terms[i-1]
                symbol2 = atom2.terms[i-1]
            if is_variable(symbol1) and is_variable(symbol2):
                pass
            elif is_variable(symbol1):
                substitution.add((symbol1, symbol2))
            elif is_variable(symbol2):
                substitution.add((symbol2, symbol1))
            else:
                score = tf.minimum(score,
                                   tf.exp(-tf.reduce_sum(
                                       (self.__embeddings[symbol1]- self.__embeddings[symbol2])**2)))
                """
                score = score*tf.exp(-tf.reduce_sum(
                                       (self.__embeddings[symbol1]- self.__embeddings[symbol2])**2))
                """
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
        or_states = self.apply_rules(NeuralProver.substitute(body[0], state.substitution),
                                     depth-1, state)
        for or_state in or_states:
            states.extend(self.apply_rule(body[1:],depth,or_state))
        return states

    def loss(self, positive, negative, depth):
        positive_loss = [-tf.log(self.prove(atom, depth)+1e-5) for atom in positive]
        negative_loss = [-tf.log(1 - self.prove(atom, depth)+1e-5) for atom in negative]
        return tf.reduce_mean(tf.stack(positive_loss+negative_loss))

    def grad(self, positive, negative, depth):
        with tfe.GradientTape() as tape:
            loss_value = self.loss(positive, negative, depth)
            weight_decay = 0.01
            regularization = 0
            for weights in self.__embeddings.variables:
                weights = tf.nn.softmax(weights)
                regularization += tf.reduce_sum(tf.sqrt(weights))*weight_decay
            loss_value += regularization/len(self.__embeddings.variables)
        return tape.gradient(loss_value, self.__embeddings.variables)

    def train(self, positive, negative, depth, steps):
        losses = []
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        for i in range(steps):
            grads = self.grad(positive, negative, depth)
            optimizer.apply_gradients(zip(grads, self.__embeddings.variables),
                                      global_step=tf.train.get_or_create_global_step())
            loss_avg = self.loss(positive, negative, depth)
            losses.append(float(loss_avg.numpy()))
            print("-"*20)
            print("step "+str(i)+" loss is "+str(loss_avg))



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
        for predicate in predicates.union(para_predicates):
            self.embbedings[predicate] = tf.get_variable(predicate.name,shape=[dimension],dtype=tf.float32)
        for constant in constants:
            self.embbedings[constant] = tf.get_variable(constant,shape=[dimension],dtype=tf.float32)

    def __getitem__(self, key):
        return self.embbedings[key]

    @property
    def variables(self):
        return self.embbedings.values()

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
    para_clauses = []
    embeddings = Embeddings.from_clauses(clauses, para_clauses)
    ntp = NeuralProver(clauses, embeddings)
    score = ntp.prove(str2atom("grandFatherOf(abe,cart)"),2)
    assert float(score) == 1.0

    clause_str = ["fatherOf(abe, homer)","parentOf(homer,cart)"]
    para_clauses = [str2clause("grandFatherOf(X,Y):-p(X,Z),q(Z,Y)")]
    clauses = [str2clause(s) for s in clause_str]
    positive = [str2atom("grandFatherOf(abe,cart)")]
    negative = [str2atom("grandFatherOf(cart,abe)"), str2atom("grandFatherOf(abe,homer)"),
                str2atom("grandFatherOf(homer,cart)"), str2atom("grandFatherOf(cart,homer)")]
    embeddings = Embeddings.from_clauses(clauses, para_clauses)
    ntp = NeuralProver(clauses+para_clauses, embeddings)
    ntp.train(positive,negative,2,500)
    score = ntp.prove(str2atom("grandFatherOf(abe,cart)"),2)
    score2 = ntp.prove(str2atom("grandFatherOf(cart,abe)"),2)
    score3 = ntp.prove(str2atom("grandFatherOf(abe,homer)"),2)
    score4 = ntp.prove(str2atom("grandFatherOf(homer,cart)"),2)
    score5 = ntp.prove(str2atom("grandFatherOf(cart,homer)"),2)
    score6 = ntp.prove(str2atom("grandFatherOf(homer,abe)"),2)
    similarity = ntp.unify(str2atom("p(abe,cart)"), str2atom("fatherOf(abe,cart)"), ProofState(set(), 1))
    similarity2 = ntp.unify(str2atom("q(abe,cart)"), str2atom("parentOf(abe,cart)"), ProofState(set(), 1))
    similarity3 = ntp.unify(str2atom("p(abe,cart)"), str2atom("parentOf(abe,cart)"), ProofState(set(), 1))
    a1 = ntp.apply_rule([str2atom("p(abe,homer)"), str2atom("q(homer,cart)")], 2, ProofState(set(), 1))
    a2 = ntp.apply_rules(str2atom("grandFatherOf(abe,cart)"), 2, ProofState(set(), 1))
    a3 = ntp.apply_rules(str2atom("grandFatherOf(abe,cart)"), 2, ProofState(set(), 1))
    a4 = ntp.apply_rules(str2atom("grandFatherOf(abe,cart)"), 2, ProofState(set(), 1))


    assert float(score) == 1.0


