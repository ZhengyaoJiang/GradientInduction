from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf
from collections import OrderedDict
import pandas as pd
import tensorflow.contrib.eager as tfe
import os
from core.rules import RulesManager
from core.clause import Predicate
from pprint import pprint

tf.enable_eager_execution()

class Agent(object):
    def __init__(self, rules_manager, ilp):
        self.rules_manager = rules_manager
        self.rule_weights = OrderedDict() # dictionary from predicates to rule weights matrices
        self.__init__rule_weights()
        self.ground_atoms = rules_manager.all_grounds
        self.base_valuation = self.axioms2valuation(ilp.background)
        self.training_data = OrderedDict() # index to label
        self.__init_training_data(ilp.positive, ilp.negative)

    def __init__rule_weights(self):
        with tf.variable_scope("rule_weights", reuse=tf.AUTO_REUSE):
            for predicate, clauses in self.rules_manager.all_clauses.items():
                self.rule_weights[predicate] = []
                for i in range(len(clauses)):
                    self.rule_weights[predicate].append(tf.get_variable(predicate.name+"_rule_weights"+str(i),
                                                                [len(clauses[i]),],
                                                                initializer=tf.random_normal_initializer,
                                                                dtype=tf.float32))

    def show_definition(self):
        for predicate, clauses in self.rules_manager.all_clauses.items():
            rules_weights = self.rule_weights[predicate]
            print(str(predicate))
            for i, rule_weights in enumerate(rules_weights):
                weights = tf.nn.softmax(rule_weights)
                indexes = np.nonzero(weights>0.05)[0]
                print("clasue {}".format(i))
                for j in range(len(indexes)):
                    print("weight is {}".format(weights[indexes[j]]))
                    print(str(clauses[i][indexes[j]]))
                print("\n")

    def __init_training_data(self, positive, negative):
        for i, atom in enumerate(self.ground_atoms):
            if atom in positive:
                self.training_data[i] = 1.0
            elif atom in negative:
                self.training_data[i] = 0.0


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

    def valuation2atoms(self, valuation):
        result = {}
        for i, value in enumerate(valuation):
            if value > 0.01:
                result[self.ground_atoms[i]] = float(value)
        return result

    def deduction(self):
        # takes background as input and return a valuation of target ground atoms
        valuation = self.base_valuation
        for _ in range(self.rules_manager.program_template.forward_n):
            valuation = self.inference_step(valuation)
        return valuation

    def inference_step(self, valuation):
        deduced_valuation = tf.zeros(len(self.ground_atoms))
        # deduction_matrices = self.rules_manager.deducation_matrices[predicate]
        for predicate, matrix in self.rules_manager.deduction_matrices.items():
            deduced_valuation += Agent.inference_single_predicate(valuation, matrix, self.rule_weights[predicate])
        return deduced_valuation+valuation - deduced_valuation*valuation

    @staticmethod
    def inference_single_predicate(valuation, deduction_matrices, rule_weights):
        '''

        :param valuation:
        :param deduction_matrices: list of list of matrices
        :param rule_weights: list of tensor, shape (number_of_rule_temps, number_of_clauses_generated)
        :return:
        '''
        result_valuations = [[] for _ in rule_weights]
        for i in range(len(result_valuations)):
            for matrix in deduction_matrices[i]:
                result_valuations[i].append(Agent.inference_single_clause(valuation, matrix))

        c_p = None
        for i in range(len(result_valuations)):
            valuations = tf.stack(result_valuations[i])
            prob_rule_weights = tf.nn.softmax(rule_weights[i])[:, None]
            if c_p==None:
                c_p = tf.reduce_sum(prob_rule_weights*valuations, axis=0)
            else:
                c_p = prob_sum(c_p, tf.reduce_sum(prob_rule_weights*valuations, axis=0))
        return c_p

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

    def loss(self, batch_size=-1):
        labels = np.array(self.training_data.values(), dtype=np.float32)
        outputs = tf.gather(self.deduction(), np.array(self.training_data.keys(), dtype=np.int32))
        if batch_size>0:
            index = np.random.randint(0, len(labels), batch_size)
            labels = labels[index]
            outputs = tf.gather(outputs, index)
        loss = -tf.reduce_mean(labels*tf.log(outputs+1e-10)+(1-labels)*tf.log(1-outputs+1e-10))
        return loss

    def grad(self):
        with tfe.GradientTape() as tape:
            loss_value = self.loss(3)
            weight_decay = 0.0
            regularization = 0
            for weights in self.__all_variables():
                weights = tf.nn.softmax(weights)
                regularization += tf.reduce_sum(tf.sqrt(weights))*weight_decay
            loss_value += regularization/len(self.__all_variables())
        return tape.gradient(loss_value, self.__all_variables())

    def __all_variables(self):
        return [weight for weights in self.rule_weights.values() for weight in weights]

    def train(self, steps=6000, name="fizz06"):
        str2weights = {str(key)+str(i):value[i] for key,value in self.rule_weights.items() for i in range(len(value))}
        checkpoint = tfe.Checkpoint(**str2weights)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.5)
        checkpoint_dir = "./model/"+name
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        losses = []

        try:
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        except Exception as e:
            print(e)
        for i in range(steps):
            grads = self.grad()
            optimizer.apply_gradients(zip(grads, self.__all_variables()),
                                      global_step=tf.train.get_or_create_global_step())
            loss_avg = self.loss()
            losses.append(float(loss_avg.numpy()))
            print("-"*20)
            print("step "+str(i)+" loss is "+str(loss_avg))
            if i%5==0:
                self.show_definition()
                for atom, value in self.valuation2atoms(self.deduction()).items():
                    print(str(atom)+": "+str(value))
                checkpoint.save(checkpoint_prefix)
            print("-"*20+"\n")
            pd.Series(losses).to_csv(name+".csv")

def prob_sum(x, y):
    return x + y - x*y

class RLAgent(Agent):
    pass
