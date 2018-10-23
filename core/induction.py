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
    def __init__(self, rules_manager, background, independent_clause=True, scope_name="rule_weights"):
        self.rules_manager = rules_manager
        self.independent_clause = independent_clause
        self.rule_weights = OrderedDict() # dictionary from predicates to rule weights matrices
        self.__init__rule_weights(scope_name)
        self.ground_atoms = rules_manager.all_grounds
        self.base_valuation = self.axioms2valuation(background)

    def __init__rule_weights(self, scope_name="rule_weights"):
        if self.independent_clause:
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                for predicate, clauses in self.rules_manager.all_clauses.items():
                    self.rule_weights[predicate] = []
                    for i in range(len(clauses)):
                        self.rule_weights[predicate].append(tf.get_variable(predicate.name+"_rule_weights"+str(i),
                                                                    [len(clauses[i]),],
                                                                    initializer=tf.random_normal_initializer,
                                                                    dtype=tf.float32))
        else:
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
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

    def deduction(self, state=None):
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

    def all_variables(self):
        if self.independent_clause:
            return [weight for weights in self.rule_weights.values() for weight in weights]
        else:
            return [weights for weights in self.rule_weights.values()]



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

    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.loss(-1)
            weight_decay = 0.0
            regularization = 0
            for weights in self.all_variables():
                weights = tf.nn.softmax(weights)
                regularization += tf.reduce_sum(tf.sqrt(weights))*weight_decay
            loss_value += regularization/len(self.all_variables())
        return tape.gradient(loss_value, self.all_variables()), loss_value

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
            grads, loss = self.grad()
            optimizer.apply_gradients(zip(grads, self.all_variables()),
                                      global_step=tf.train.get_or_create_global_step())
            loss_avg = float(loss.numpy())
            losses.append(loss_avg)
            print("-"*20)
            print("step "+str(i)+" loss is "+str(loss_avg))
            if i%5==0:
                self.show_definition()
                valuation_dict = self.valuation2atoms(self.deduction()).items()
                for atom, value in valuation_dict:
                    print(str(atom)+": "+str(value))
                if name:
                    checkpoint.save(checkpoint_prefix)
                    pd.Series(np.array(losses)).to_csv(name+".csv")
            print("-"*20+"\n")
        return losses

class RLDILP(BaseDILP):
    def __init__(self, rules_manager, env, independent_clause=True):
        super(RLDILP, self).__init__(rules_manager, env.background, independent_clause)
        self.env = env

    def valuation2action_prob(self, valuation, state):
        """
        :param valuation:
        :param state: tuple of terms
        :return: action probabilities, difference between sum of action probabilities and 1
        """
        atoms = self.valuation2atoms(valuation, -1).keys() #ordered
        indexes = [None for _ in self.env.actions]
        for i,atom in enumerate(atoms):
            if state == atom.terms and atom.predicate in self.env.actions:
                indexes[self.env.actions.index(atom.predicate)] = i
        action_eval = tf.gather(valuation, indexes)
        sum_action_eval = tf.reduce_sum(action_eval)
        if sum_action_eval>1.0:
            action_prob = action_eval/sum_action_eval
        else:
            action_prob = action_eval + (1.0-sum_action_eval)/len(self.env.actions)
        return action_prob, sum_action_eval-1.0

    def get_str2weights(self):
        if self.independent_clause:
            str2weights = {str(key) + str(i): value[i] for key, value in self.rule_weights.items() for i in
                           range(len(value))}
        else:
            str2weights = {str(key): value for key, value in self.rule_weights.items()}
        return str2weights

    def create_checkpoint(self, name):
        if name:
            checkpoint = tfe.Checkpoint(**self.get_str2weights())
            checkpoint_dir = "./model/"+name
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            try:
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            except Exception as e:
                print(e)
            return checkpoint, checkpoint_prefix
        else:
            return None, None


    def log(self):
        self.show_definition()
        valuation_dict = self.valuation2atoms(self.deduction()).items()
        for atom, value in valuation_dict:
            print(str(atom)+": "+str(value))




class ReinforceLearner(object):
    def __init__(self, agent, enviornment):
        # super(ReinforceDILP, self).__init__(rules_manager, enviornment.background)
        self.env = enviornment
        self.agent = agent

    def create_checkpoint(self, optimizer):
        model_parameters = {}
        if isinstance(self.agent, RLDILP):
            model_parameters.update(self.agent.get_str2weights())
        elif isinstance(self.agent, NeuralAgent):
            model_parameters["actor"] = self.agent.model
        root = tf.train.Checkpoint(optimizer=optimizer,
                                   optimizer_step=tf.train.get_or_create_global_step(),
                                   **model_parameters)
        return root


    def sample_episode(self):
        if isinstance(self.agent, RLDILP):
            valuation = self.agent.deduction(self.env.state)
        action_prob_history = []
        action_history = []
        reward_history = []
        action_trajectory_prob = []
        state_history = []
        excesses = []
        while True:
            if isinstance(self.agent, RLDILP):
                action_prob, excess = self.agent.valuation2action_prob(valuation, self.env.state)
            else:
                action_prob = self.agent.deduction(self.env.state)
                excess = 0
            excesses.append(excess)
            action_index = np.random.choice(range(len(self.env.actions)), p=action_prob.numpy())
            action = self.env.action_index2symbol(action_index)
            reward, finished = self.env.next_step(action)
            state_history.append(self.env.state)
            reward_history.append(reward)
            action_history.append(action_index)
            action_trajectory_prob.append(action_prob[action_index])
            if finished:
                self.env.reset()
                break
        total_excess = tf.reduce_sum(tf.stack(excesses)**2)
        return reward_history, action_history, action_trajectory_prob, state_history, total_excess

    def loss(self, selected_action_prob, returns, additional_discount):
        return -tf.log(tf.clip_by_value(selected_action_prob, 1e-5, 1.0))*returns*additional_discount

    def train(self, steps=300, name=None, discounting=1.0, batched=True, learning_rate=0.5):
        losses = []
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        checkpoint_dir = "./model/" + name
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = self.create_checkpoint(optimizer)
        try:
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        except Exception as e:
            print(e)

        for i in range(steps):
            if batched:
                with tf.GradientTape() as tape:
                    reward_history, action_history, action_prob_history, state_history, excess = \
                        self.sample_episode()
                    returns = discount(reward_history, discounting)
                    additional_discount = np.cumprod(discounting*np.ones_like(returns))
                    log = {"return":returns[0], "action_history":[self.env.action_index2symbol(action_index).name for action_index in action_history]}
                    loss_value = tf.reduce_sum(self.loss(action_prob_history, returns, additional_discount))
                    #loss_value += 1.0*excess
                grads = tape.gradient(loss_value, self.agent.all_variables())
                optimizer.apply_gradients(zip(grads, self.agent.all_variables()),
                                          global_step=tf.train.get_or_create_global_step())
            else:
                reward_history, action_history, action_prob_history, state_history, excess\
                    = self.sample_episode()
                returns = discount(reward_history, discounting)
                additional_discount = np.cumprod(discounting*np.ones_like(returns))
                log = {"return":returns[0], "action_history":[self.env.action_index2symbol(action_index).name for action_index in action_history]}
                for action_index, r, acc_discount, s in zip(action_history, returns, additional_discount, state_history):
                    with tf.GradientTape() as tape:
                        if isinstance(self.agent, RLDILP):
                            valuation = self.agent.deduction()
                            prob = self.agent.valuation2action_prob(valuation, s)[action_index]
                        else:
                            prob = self.agent.deduction(s)[action_index]
                        loss_value = self.loss(prob, r, acc_discount)
                    grads = tape.gradient(loss_value, self.agent.all_variables())
                    optimizer.apply_gradients(zip(grads, self.agent.all_variables()),
                                          global_step=tf.train.get_or_create_global_step())
            print("-"*20)
            print("step "+str(i)+"return is "+str(log["return"]))
            if i%5==0:
                self.agent.log()
                pprint(log)
                if name:
                    checkpoint.save(checkpoint_prefix)
                    pd.Series(np.array(losses)).to_csv(name+".csv")
            print("-"*20+"\n")
        return log["return"]

class PPOLearner(ReinforceLearner):
    def __init__(self, agent, enviornment, critic=None):
        super(PPOLearner, self).__init__(agent, enviornment)
        self.epsilon = 0.1
        self.critic = critic
        self.state_size = len(enviornment.state)
        self.discounting = 1.0

    def create_checkpoint(self, optimizer):
        model_parameters = {}
        if self.critic:
            model_parameters["critic"]=self.critic.model
        if isinstance(self.agent, RLDILP):
            model_parameters.update(self.agent.get_str2weights())
        elif isinstance(self.agent, NeuralAgent):
            model_parameters["actor"] = self.agent.model

        root = tf.train.Checkpoint(optimizer=optimizer,
                                   optimizer_step=tf.train.get_or_create_global_step(),
                                   **model_parameters)
        return root


    def actor_loss(self, old_prob, new_prob, advantage):
        ratio = new_prob / old_prob
        return -tf.reduce_sum(tf.minimum(ratio*advantage,
                                          tf.clip_by_value(ratio, 1.-self.epsilon, 1.+self.epsilon)*advantage))

    def entropy_loss(self, action_probs):
        entropy = -action_probs*tf.log(tf.clip_by_value(action_probs, 1e-5, 1.0))
        return -tf.reduce_sum(entropy)

    def critic_loss(self, reward, current_state_value, next_state_value):
        td_error = reward - current_state_value + self.discounting*next_state_value
        loss = tf.square(td_error)
        return tf.reduce_sum(loss)


    def get_action_prob(self, states, action_indexes):
        action_probs = []
        all_action_probs = []
        if isinstance(self.agent, RLDILP):
            valuation = self.agent.deduction(self.env.state)
        for state, action_index in zip(states, action_indexes):
            if isinstance(self.agent, RLDILP):
                action_prob,_ = self.agent.valuation2action_prob(valuation, state)
            else:
                action_prob = self.agent.deduction(state)
            action_probs.append(action_prob[action_index])
            all_action_probs.append(action_prob)
        return tf.stack(action_probs), tf.stack(all_action_probs)

    def generate_value(self, states):
        """
        :param states:
        :return:
        """
        current_value = []
        future_value = []
        for episode in states:
            values = self.critic.batch_predict(episode)
            future_v = np.concatenate([values.numpy()[1:], [0.0]])
            current_value.append(values)
            future_value.append(future_v)
        return tf.concat(current_value, axis=0), np.concatenate(future_value)

    def summary_scalar(self, name, scalar):
        if self.summary_writer:
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar(name, scalar)


    def train(self, steps=300, name=None, learning_rate=0.5, critic_ratio=0.003):
        losses = []
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        checkpoint_dir = "./model/" + name
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = self.create_checkpoint(optimizer)
        try:
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        except Exception as e:
            print(e)

        self.summary_writer = tf.contrib.summary.create_file_writer("./model/"+name, flush_millis=10000)

        for i in range(steps):
            reward_history, action_history, action_prob_history, state_history = [], [], [], []
            additional_discount = None
            returns = None
            for j in range(10):
                r, a, a_p, s, _ =  self.sample_episode()
                rs = discount(r, self.discounting)
                a_dis = np.cumprod(self.discounting * np.ones_like(rs))
                reward_history += r
                action_history += a
                action_prob_history += a_p
                state_history.append(s)
                returns = np.concatenate([returns, rs]) if isinstance(returns,np.ndarray) else rs
                self.summary_scalar("return", rs[0])

                additional_discount = np.concatenate([additional_discount, a_dis]) \
                    if isinstance(additional_discount,np.ndarray) else a_dis
            for j in range(10):
                with tf.GradientTape() as tape:
                    log = {"return": rs[0],
                           "action_history": [self.env.action_index2symbol(action_index).name for action_index in
                                              a]}
                    new_action_prob, all_action_prob = self.get_action_prob(sum(state_history, []),
                                                                            action_history)
                    entropy_loss = self.entropy_loss(all_action_prob)
                    self.summary_scalar("entropy_loss", entropy_loss)
                    if self.critic:
                        current_value, next_value = self.generate_value(state_history)
                        loss_value = self.actor_loss(action_prob_history, new_action_prob,
                                                 (reward_history-current_value.numpy()+next_value*self.discounting))
                        self.summary_scalar("actor_loss", loss_value)
                        critic_loss = self.critic_loss(tf.stack(reward_history),
                                                       current_value,
                                                       next_value)
                        self.summary_scalar("critic_loss", critic_loss)
                        loss_value += critic_loss*critic_ratio
                        all_variables = self.agent.all_variables()+self.critic.all_variables()

                    else:
                        loss_value = self.actor_loss(action_prob_history, new_action_prob,
                                                 returns*additional_discount)
                        self.summary_scalar("actor_loss", loss_value)
                        all_variables = self.agent.all_variables()
                    loss_value += 0.1*entropy_loss
                    # loss_value += 1.0*excess
                self.summary_scalar("total_loss", loss_value)
                grads = tape.gradient(loss_value, all_variables)
                optimizer.apply_gradients(zip(grads, all_variables),
                                              global_step=tf.train.get_or_create_global_step())
            print("-" * 20)
            print("step " + str(i) + "return is " + str(log["return"]))
            if i % 1 == 0:
                self.agent.log()
                pprint(log)
                if name:
                    checkpoint.save(checkpoint_prefix)
                    pd.Series(np.array(losses)).to_csv(name + ".csv")
            print("-" * 20 + "\n")
        return log["return"]


class NeuralAgent(object):
    def __init__(self, unit_list, action_size, state_size):
        self.unit_list = unit_list
        self.action_size = action_size
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(unit_list[0], input_shape=(state_size,), activation=tf.nn.relu,
                                        kernel_initializer=tf.initializers.random_normal()))
        for i in range(1, len(unit_list)):
            model.add(tf.keras.layers.Dense(unit_list[i], activation=tf.nn.relu,
                                            kernel_initializer=tf.initializers.random_normal()))
        model.add(tf.keras.layers.Dense(action_size, activation=tf.nn.softmax))
        self.model=model

    def deduction(self, state):
        inputs = np.array([[float(s) for s in state]])
        return self.model(inputs)[0]

    def batch_deduction(self, states):
        inputs = np.array([[float(s) for s in state] for state in states])
        return self.model(inputs)[:,0]

    def all_variables(self):
        return self.model.variables

    def log(self):
        pass

class NeuralCritic(object):
    def __init__(self, unit_list, state_size):
        self.unit_list = unit_list
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(unit_list[0], kernel_initializer=tf.initializers.random_normal()
                  , input_shape=(state_size,), activation=tf.nn.relu))
        for i in range(1, len(unit_list)):
            model.add(tf.keras.layers.Dense(unit_list[i], kernel_initializer=tf.initializers.random_normal(),
                                            activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(1))
        self.model=model

    def predict(self, state):
        inputs = np.array([[float(s) for s in state]])
        return self.model(inputs)[0]

    def batch_predict(self, states):
        inputs = np.array([[float(s) for s in state] for state in states])
        return self.model(inputs)[:, 0]

    def all_variables(self):
        return self.model.variables



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
