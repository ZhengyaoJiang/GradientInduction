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

class BaseDILP(object):
    def __init__(self, rules_manager, background, independent_clause=True):
        self.rules_manager = rules_manager
        self.independent_clause = independent_clause
        self.rule_weights = OrderedDict() # dictionary from predicates to rule weights matrices
        self.__init__rule_weights()
        self.ground_atoms = rules_manager.all_grounds
        self.base_valuation = self.axioms2valuation(background)

    def __init__rule_weights(self):
        if self.independent_clause:
            with tf.variable_scope("rule_weights", reuse=tf.AUTO_REUSE):
                for predicate, clauses in self.rules_manager.all_clauses.items():
                    self.rule_weights[predicate] = []
                    for i in range(len(clauses)):
                        self.rule_weights[predicate].append(tf.get_variable(predicate.name+"_rule_weights"+str(i),
                                                                    [len(clauses[i]),],
                                                                    initializer=tf.random_normal_initializer,
                                                                    dtype=tf.float32))
        else:
            with tf.variable_scope("rule_weights", reuse=tf.AUTO_REUSE):
                for predicate, clauses in self.rules_manager.all_clauses.items():
                    self.rule_weights[predicate] = tf.get_variable(predicate.name + "_rule_weights",
                                                                   [len(clauses[0]), len(clauses[1])],
                                                                   initializer=tf.random_normal_initializer,
                                                                   dtype=tf.float32)

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

    def valuation2atoms(self, valuation, threshold=0.5):
        result = OrderedDict()
        for i, value in enumerate(valuation):
            if value >= threshold:
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
            deduced_valuation += BaseDILP.inference_single_predicate(valuation, matrix, self.rule_weights[predicate])
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
                result_valuations[i].append(BaseDILP.inference_single_clause(valuation, matrix))

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

    def grad(self):
        with tf.GradientTape() as tape:
            loss_value, log = self.loss(-1)
            weight_decay = 0.0
            regularization = 0
            for weights in self.__all_variables():
                weights = tf.nn.softmax(weights)
                regularization += tf.reduce_sum(tf.sqrt(weights))*weight_decay
            loss_value += regularization/len(self.__all_variables())
        return tape.gradient(loss_value, self.__all_variables()), loss_value, log

    def __all_variables(self):
        if self.independent_clause:
            return [weight for weights in self.rule_weights.values() for weight in weights]
        else:
            return [weights for weights in self.rule_weights.values()]

    def train(self, steps=300, name=None):
        """
        :param steps:
        :param name:
        :return: the loss history
        """
        if self.independent_clause:
            str2weights = {str(key) + str(i): value[i] for key, value in self.rule_weights.items() for i in
                           range(len(value))}
        else:
            str2weights = {str(key): value for key, value in self.rule_weights.items()}

        if name:
            checkpoint = tfe.Checkpoint(**str2weights)
            checkpoint_dir = "./model/"+name
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            try:
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            except Exception as e:
                print(e)

        losses = []
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.5)

        for i in range(steps):
            grads, loss, log = self.grad()
            optimizer.apply_gradients(zip(grads, self.__all_variables()),
                                      global_step=tf.train.get_or_create_global_step())
            loss_avg = float(loss.numpy())
            losses.append(loss_avg)
            print("-"*20)
            print("step "+str(i)+" loss is "+str(loss_avg))
            if i%5==0:
                self.show_definition()
                valuation_dict = self.valuation2atoms(self.deduction()).items()
                pprint(log)
                for atom, value in valuation_dict:
                    print(str(atom)+": "+str(value))
                if name:
                    checkpoint.save(checkpoint_prefix)
                    pd.Series(np.array(losses)).to_csv(name+".csv")
            print("-"*20+"\n")
        return losses


class SupervisedDILP(BaseDILP):
    def __init__(self, rules_manager, ilp):
        super(SupervisedDILP, self).__init__(rules_manager, ilp.background)
        self.training_data = OrderedDict() # index to label
        self.__init_training_data(ilp.positive, ilp.negative)

    def __init_training_data(self, positive, negative):
        for i, atom in enumerate(self.ground_atoms):
            if atom in positive:
                self.training_data[i] = 1.0
            elif atom in negative:
                self.training_data[i] = 0.0

    def loss(self, batch_size=-1):
        labels = np.array(self.training_data.values(), dtype=np.float32)
        outputs = tf.gather(self.deduction(), np.array(self.training_data.keys(), dtype=np.int32))
        if batch_size>0:
            index = np.random.randint(0, len(labels), batch_size)
            labels = labels[index]
            outputs = tf.gather(outputs, index)
        loss = -tf.reduce_mean(labels*tf.log(outputs+1e-10)+(1-labels)*tf.log(1-outputs+1e-10))
        return loss



class ReinforceDILP(BaseDILP):
    def __init__(self, rules_manager, enviornment):
        super(ReinforceDILP, self).__init__(rules_manager, enviornment.background)
        self.env = enviornment

    def valuation2action_prob(self, valuation, state):
        """
        :param valuation:
        :param state: tuple of terms
        :return:
        """
        atoms = self.valuation2atoms(valuation, -1).keys() #ordered
        indexes = [None for _ in self.env.actions]
        for i,atom in enumerate(atoms):
            if state == atom.terms and atom.predicate in self.env.actions:
                indexes[self.env.actions.index(atom.predicate)] = i
        action_eval = tf.gather(valuation, indexes)
        action_prob = action_eval / tf.reduce_sum(action_eval)
        return action_prob

    def sample_episode(self):
        valuation = self.deduction()
        action_prob_history = []
        action_history = []
        reward_history = []
        action_trajectory_prob = []
        while True:
            action_prob = self.valuation2action_prob(valuation, self.env.state)
            action_index = np.random.choice(range(len(self.env.actions)), p=action_prob.numpy())
            action = self.env.action_index2symbol(action_index)
            reward, finished = self.env.step(action)
            reward_history.append(reward)
            action_history.append(action_index)
            action_prob_history.append(action_prob)
            action_trajectory_prob.append(action_prob[action_index])
            if finished:
                self.env.reset()
                break
        return reward_history, action_history, action_prob_history, action_trajectory_prob

    def loss(self, batch_size=-1, discounting=0.95):
        #TODO: enable discounting here
        reward_history, action_history, action_prob_history, action_trajectory_prob = self.sample_episode()
        reward_history = tf.stack(reward_history)
        action_trajectory_prob = tf.stack(action_trajectory_prob)
        returns = discount(reward_history.numpy(), discounting)
        additional_discount = np.cumprod(discounting*np.ones_like(returns))
        log = {"return":returns[0], "action_history":[self.env.action_index2symbol(action_index).name for action_index in action_history]}

        return -tf.reduce_sum(tf.log(action_trajectory_prob)*returns*additional_discount), log

def discount(r, discounting, normal=False):
    discounted_reward = np.zeros_like(r)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * discounting + r[i]
        discounted_reward[i] = G
    if normal:
        mean = np.mean(discounted_reward)
        std = np.std(discounted_reward)
        discounted_reward = (discounted_reward - mean) / (std)
    return discounted_reward

def prob_sum(x, y):
    return x + y - x*y
