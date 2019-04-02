from core.induction import *
from core.NTP import *
from copy import deepcopy
from collections import namedtuple
import tensorflow as tf
import pickle
import json

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

Episode = namedtuple("Episode", ["reward_history", "action_dist",
                                 "action_history", "action_trajectory_prob", "state_history",
                                 "input_vector_history",
                                 "returns", "steps", "advantages", "final_return"])

class ReinforceLearner(object):
    def __init__(self, agent, enviornment, learning_rate, critic=None,
                 steps=300, name=None, discounting=1.0, batched=True, optimizer="RMSProp",
                 end_by_episode=True,
                 minibatch_size=1, gradient_noise_rate=0.0,
                 gradient_noise_exp=0.55,
                 log_steps=100):
        # super(ReinforceDILP, self).__init__(rules_manager, enviornment.background)
        if isinstance(agent, HybridAgent):
            self.type = "Hybrid"
        elif isinstance(agent, NTPAgent):
            self.type = "NTP"
        elif isinstance(agent, NeuralAgent):
            self.type = "NN"
        else:
            self.type = "Random"
        self.env = enviornment
        self.agent = agent
        self.state_encoding = agent.state_encoding
        self.learning_rate = learning_rate
        self.gradient_noise_rate = gradient_noise_rate
        self.gradient_noise_exp = gradient_noise_exp
        self.critic=critic
        self.total_steps = steps
        self.name = name
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.discounting = discounting
        self.batched = batched
        self.end_by_episode=end_by_episode
        self.batch_size = minibatch_size
        self.log_steps = log_steps
        self.dynamic_records = []

    def train(self, state_history, advantage, action_index):
        #advantage = np.array(advantage)
        with tf.GradientTape() as tape:
            #tape.watch(self.agent.all_variables())
            if self.batched:
                action_dist,_ = self.decide(state_history)
                indexed_action_prob = tf.batch_gather(action_dist,
                                                      tf.convert_to_tensor(action_index)[:, None])[:, 0]
                loss = self.loss(indexed_action_prob, advantage)
        gradients = tape.gradient(loss, self.agent.all_variables())
        step = tf.train.get_or_create_global_step()
        if self.gradient_noise_rate>0:
            noisy_gradient = []
            std = self.gradient_noise_rate / tf.pow((1.0+tf.cast(step, dtype=tf.float32)),
                                                    tf.constant(self.gradient_noise_exp,
                                                                            dtype=tf.float32))
            for gradient in gradients:
                noisy_gradient.append(gradient+
                                      tf.random_normal(tf.shape(gradient),
                                                       stddev=std))
            gradients = noisy_gradient
        try:
            self.optimizer.apply_gradients(zip(gradients, self.agent.all_variables()), global_step=step)
        except Exception as e:
            # For random agent
            if self.type != "Random":
                raise e
        #self.saver = tf.train.Saver()

    def loss(self, indexed_action_prob, advantage):
        rl_loss = (-tf.reduce_sum(tf.log(tf.clip_by_value(indexed_action_prob, 1e-5, 1.0))
               )*advantage)
        #excess_penalty = 0.01*tf.reduce_sum(tf.nn.relu(tf.reduce_sum(self.tf_action_eval, axis=1)-1.0)**2)
        #regularization_loss = 1e-4*tf.reduce_mean(tf.stack([tf.nn.l2_loss(v) for v in self.agent.all_variables()]))
        #entropy_loss = tf.reduce_sum(self.tf_action_prob*tf.log(self.tf_action_prob))
        return rl_loss#+regularization_loss

    def decide(self, states):
        if self.type == "NTP":
            inputs = None # inputs are needed only for neural network models, so this is none
            action_prob = [self.agent.decide(list(self.env.state2atoms(states[0])))]
        elif self.type == "NN":
            inputs = np.array([self.env.state2vector(state) for state in states], dtype=np.float32)
            action_prob = self.agent.decide(inputs, False)
        elif self.type == "Random":
            inputs = None
            action_prob = [np.ones([self.env.action_n])/ self.env.action_n for _ in states]
        else:
            raise ValueError()
        return action_prob, inputs

    def sample_episode(self, max_steps=99999):
        action_dist = []
        action_history = []
        reward_history = []
        action_trajectory_prob = []
        state_history = []
        input_vector_history = []
        steps = []
        step = 0
        while step<max_steps:
            action_prob, inputs = self.decide([self.env.state])
            action_prob = action_prob[0].numpy()
            action_index = np.random.choice(range(self.env.action_n), p=action_prob)
            if self.state_encoding == "atoms":
                action = self.env.all_actions[action_index]
            elif self.state_encoding =="vector":
                if action_index<len(self.env.all_actions):
                    action = self.env.all_actions[action_index]
                else:
                    action = np.random.choice(self.env.all_actions)
            else:
                raise ValueError()
            steps.append(step)
            state_history.append(self.env.state)
            reward, finished = self.env.next_step(action)
            reward_history.append(reward)
            action_history.append(action_index)
            action_dist.append(action_prob)
            action_trajectory_prob.append(action_prob[action_index])
            input_vector_history.append(inputs)
            step += 1
            if finished:
                self.env.reset()
                break
        final_return = [np.sum(reward_history)]
        returns = discount(reward_history, self.discounting)
        if self.critic:
            values = self.critic.get_values(state_history,steps).numpy().flatten()
            self.critic.batch_learn(state_history, reward_history, steps)
            #advantages = normalize(generalized_adv(reward_history, values, self.discounting))
            advantages = generalized_adv(reward_history, values, self.discounting)
            #advantages = np.array(returns)# - values
        else:
            advantages = returns
        advantages[-1] = 0.0
        return Episode(reward_history, action_dist, action_history, action_trajectory_prob, state_history,
                       input_vector_history, returns, steps, advantages, final_return)

    def get_minibatch_buffer(self, batch_size=50, end_by_episode=True):
        empty_buffer = [[] for _ in range(9)]
        episode_buffer = deepcopy(empty_buffer)
        sample_related_indexes = range(9)

        def dump_episode2buffer(episode):
            # will remove the final step!!!
            for i in sample_related_indexes:
                episode_buffer[i].extend(episode[i][:-1])

        def split_buffer(raw_buffer, index):
            if len(episode_buffer[0]) < index:
                return raw_buffer, deepcopy(empty_buffer)
            result = []
            new_buffer = []
            for l in raw_buffer:
                result.append(l[:index])
                new_buffer.append(l[index:])
            return result, new_buffer

        while True:
            if len(episode_buffer[0]) ==0:
                if end_by_episode:
                    e = self.sample_episode()
                    dump_episode2buffer(e)
                    final_return = e.final_return
            if not end_by_episode:
                while len(episode_buffer[0]) < batch_size:
                    e = self.sample_episode()
                    dump_episode2buffer(e)
                    final_return = e.final_return
            result, episode_buffer = split_buffer(episode_buffer, batch_size)
            yield Episode(*(result+[final_return]))


    def summary_scalar(self, name, scalar):
        if self.summary_writer:
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar(name, scalar)

    def summary_histogram(self, name, data):
        if self.summary_writer:
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.histogram(name, data)

    def setup_train(self, auto_load=True):
        if self.name:
            if auto_load:
                try:
                    path = "./model/" + self.name
                    self.load(path)
                except Exception as e:
                    print(e)
            #self.summary_writer = tf.contrib.summary.create_file_writer("./model/"+self.name, flush_millis=10000)
            #self.summary_scalar("returns", self.tf_returns[0])
            #self.summary_histogram("advantages", self.tf_advantage)
            #self.summary_scalar("loss", self.tf_loss)
            #self.summary_histogram("weights", tf.concat(self.agent.all_variables(), axis=0))
            #with self.summary_writer.as_default():
            #    tf.contrib.summary.initialize(graph=tf.get_default_graph(), session=sess)
        else:
            self.summary_writer = None
            # modelArxivICML definition code goes here
            # and in it call

    def evaluate(self, repeat=200):
        results = []
        with tf.Session() as sess:
            self.setup_train(sess)
            self.agent.log(sess)
            rules = self.agent.get_predicates_definition(sess, threshold=0.05) if self.type == "DILP" else []
            for _ in range(repeat):
                e = self.sample_episode(sess)
                reward_history, action_dist, action_history, action_prob_history, state_history, \
                valuation_history, valuation_index_history, input_vector_history, returns, steps, adv, final_return = e
                results.append(final_return)
        unique, counts = np.unique(results, return_counts=True)
        distribution =  dict(zip(unique, counts))
        return {"distribution": distribution, "mean": np.mean(results), "std": np.std(results),
                "min": np.min(results), "max": np.max(results), "rules": rules}

    def train_step(self):
        e = next(self.minibatch_buffer)
        reward_history, action_dist, action_history, action_prob_history, state_history,\
            input_vector_history, returns, steps, advantage, final_return = e

        log = {"return":final_return[0], "action_history":[str(self.env.all_actions[action_index])
                                                               for action_index in action_history]}
        self.train(state_history, advantage, action_history)
        return log

    @property
    def checkpoints(self):
        state_dict = {"optimizer": self.optimizer}
        state_dict.update(self.agent.checkpoints)
        if self.critic:
            state_dict.update(self.critic.checkpoints)
        return state_dict

    def save(self, path):
        ckpt = tf.train.Checkpoint(**self.checkpoints)
        ckpt.save(path+"/ckpt/")
        #self.dynamic_records.append(self.agent.get_predicates_definition(sess))
        #with open(path+"/dynamics.json", "w") as f:
        #    json.dump(self.dynamic_records, f)

    def load(self, path):
        """
        self.saver.restore(sess, path+"/parameters.ckpt")
        if self.critic and isinstance(self.critic, TableCritic):
            self.critic.load(path + "/critic.pl")
        if self.type=="NTP":
            with open(path+"/dynamics.json", "r") as read_file:
                self.dynamic_records=json.load(read_file)
        """
        ckpt = tf.train.Checkpoint(**self.checkpoints)
        ckpt.restore(tf.train.latest_checkpoint(path+"/ckpt/"))

    def start_train(self):
        self.setup_train()
        self.minibatch_buffer = self.get_minibatch_buffer(batch_size=self.batch_size,
                                                          end_by_episode=self.end_by_episode)
        for i in range(self.total_steps):
            log = self.train_step()
            print("-"*20)
            print("step "+str(i)+"return is "+str(log["return"]))
            if i%self.log_steps==0:
                self.agent.log()
                if self.name:
                    path = "./model/" + self.name
                    self.save(path)
                pprint(log)
            print("-"*20+"\n")
        return log["return"]

class PPOLearner(ReinforceLearner):
    def __init__(self, agent, enviornment, learning_rate, critic=None,
                 steps=300, name=None, discounting=1.0, optimizer="RMSProp", target_kl=0.1):
        self.epsilon = 0.2
        self.tf_previous_action_prob = tf.placeholder(tf.float32, shape=[None])
        super(PPOLearner, self).__init__(agent, enviornment, learning_rate, critic,
                                         steps, name, discounting, batched=True, optimizer="RMSProp",
                                         end_by_episode=False, minibatch_size=100)
        self.tf_previous_action_dist = tf.placeholder(tf.float32, shape=[None, len(self.env.all_actions)])
        self.tf_kl = tf.reduce_mean(tf.log(self.tf_action_prob) - tf.log(self.tf_previous_action_prob))
        self.log_steps = 10
        self.target_kl = target_kl


    def loss(self, new_prob):
        ratio = tf.clip_by_value(new_prob, 1e-5, 1.0) / self.tf_previous_action_prob
        min_adv = tf.where(self.tf_advantage > 0, (1 + self.epsilon)*self.tf_advantage,
                           (1 - self.epsilon)*self.tf_advantage)
        return -tf.reduce_mean(tf.minimum(ratio*self.tf_advantage,min_adv))

    def entropy_loss(self, action_probs):
        entropy = -action_probs*tf.log(tf.clip_by_value(action_probs, 1e-5, 1.0))
        return -tf.reduce_sum(entropy)

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

    def train_step(self, sess):
        e = self.minibatch_buffer.next()
        #e = self.sample_episode(sess)
        reward_history, action_dist, action_history, action_prob_history, state_history,\
            valuation_history, valuation_index_history, input_vector_history,\
            returns, steps, advantage, final_return = e

        advantage = normalize(np.array(advantage))

        additional_discount = np.ones_like(advantage)
        log = {"return":final_return, "action_history":[str(self.env.all_actions[action_index])
                                                          for action_index in action_history]}

        for j in range(10):
            if j == 0 and self.name:
                ops = [self.tf_kl, self.tf_train, tf.contrib.summary.all_summary_ops()]
            else:
                ops = [self.tf_kl, self.tf_train]
            if self.type == "DILP":
                feed_dict = {self.tf_advantage:np.array(advantage),
                             self.tf_additional_discount:np.array(additional_discount),
                                 self.tf_returns:final_return,
                                 self.tf_previous_action_dist: action_dist,
                                 self.tf_previous_action_prob: np.array(action_prob_history),
                                 self.tf_action_index:np.array(action_history),
                                 self.tf_actions_valuation_indexes: np.array(valuation_index_history),
                                 self.agent.tf_input_valuation: np.array(valuation_history)}
            elif self.type == "Hybrid":
                feed_dict = {self.tf_advantage: np.array(advantage),
                         self.tf_additional_discount: np.array(additional_discount),
                         self.tf_returns: final_return,
                         self.tf_previous_action_dist: action_dist,
                         self.tf_previous_action_prob: np.array(action_prob_history),
                         self.tf_action_index: np.array(action_history),
                         self.tf_actions_valuation_indexes: np.array(valuation_index_history),
                         self.agent.tf_input: np.array(input_vector_history)}
            elif self.type == "NN":
                feed_dict = {self.tf_advantage:np.array(advantage),
                             self.tf_additional_discount:np.array(additional_discount),
                             self.tf_returns:final_return,
                             self.tf_previous_action_dist: action_dist,
                             self.tf_action_index:np.array(action_history),
                             self.agent.tf_is_training: True,
                             self.tf_previous_action_prob: np.array(action_prob_history),
                             self.agent.tf_input: np.array(input_vector_history)}
            result = sess.run(ops, feed_dict)
            if result[0] > self.target_kl:
                print("early stop")
                break
        return log

class RandomAgent(object):
    def __init__(self, action_size):
        self.tf_input = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        ones = tf.ones_like(self.tf_input)/ action_size
        self.tf_output = ones * tf.ones([1, action_size])/ action_size
        self.state_encoding = "vector"

    def all_variables(self):
        return []

    def log(self, sess):
        pass

class NeuralAgent(object):
    def __init__(self, unit_list, action_size, state_size):
        self.unit_list = unit_list
        self.action_size = action_size
        self.state_encoding = "vector"
        layers = []
        for unit_n in self.unit_list:
            layers.append(tf.keras.layers.Dense(unit_n, activation=tf.nn.leaky_relu))
        layers.append(tf.keras.layers.Dense(action_size, activation=tf.nn.softmax))
        self.model = tf.keras.Sequential(layers)

    def decide(self, inputs, is_training):
        return self.model(inputs)

    def all_variables(self):
        return self.model.variables

    @property
    def checkpoints(self):
        return {"actor_model":self.model}

    def log(self):
        pass


class NeuralCritic(object):
    def __init__(self, unit_list, state_size, discounting, learning_rate, state2vector,
                 involve_steps=True):
        self.unit_list = unit_list
        self.state2vector = state2vector
        self.involve_steps = involve_steps
        self.state_encoding = "vector"
        self.discounting = discounting
        self.learning_rate = learning_rate
        layers = []
        for unit_n in self.unit_list:
            layers.append(tf.keras.layers.Dense(unit_n, activation=tf.nn.leaky_relu))
        layers.append(tf.keras.layers.Dense(1))
        self.model = tf.keras.Sequential(layers)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def all_variables(self):
        return self.model.trainable_variables

    @property
    def checkpoints(self):
        return {"critic_model": self.model, "critic_optimizer":self.optimizer}

    def log(self, sess):
        pass

    def batch_learn(self, states, rewards, steps):
        returns = discount(rewards, self.discounting)
        with tf.GradientTape() as tape:
            outputs = self.get_values(states, steps)
            loss = tf.reduce_sum(tf.square(outputs[:, 0] - returns))
        grads = tape.gradient(loss, self.model.variables)
        self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                  global_step=tf.train.get_or_create_global_step())

    def get_values(self, states, steps=None):
        states = [self.state2vector(s) for s in states]
        steps = np.array(steps)
        inputs=np.array(states)
        if self.involve_steps:
            inputs = tf.concat([inputs, steps[:, np.newaxis]], axis=1)
        outputs = self.model(inputs)
        return outputs

class TableCritic(object):
    def __init__(self, discounting, learning_rate=0.1, involve_steps=False):
        self.__table = {}
        self.__discounting = discounting
        self.__learning_rate = learning_rate
        self.involve_steps = involve_steps

    def batch_learn(self, states, rewards, sess=None):
        for s, a, s2, step in zip(states, rewards, states[1:]+["end"], range(len(rewards))):
            if self.involve_steps:
                self.learn((s, step), a, (s2, step+1))
            else:
                self.learn(s, a, s2)

    def get_values(self, states, sess=None, steps=None):
        for i,state in enumerate(states):
            states[i] = totuple(state) if isinstance(state, np.ndarray) or isinstance(state, list) else state
        if self.involve_steps:
            return np.array([self.__table[(state, step)] for step,state in zip(steps, states)])
        else:
            return np.array([self.__table[state] for step,state in enumerate(states)])

    def save(self, path):
        with open(path, "w") as fh:
            pickle.dump(self.__table, fh)

    def load(self, path):
        with open(path) as fh:
            self.__table = pickle.load(fh)

    def learn(self, state, reward, next_state):
        state = totuple(state) if isinstance(state, np.ndarray) or isinstance(state, list) else state
        next_state = totuple(next_state) if isinstance(next_state, np.ndarray) or isinstance(next_state, list) else next_state
        if state not in self.__table:
            self.__table[state] = 0
        if next_state not in self.__table:
            self.__table[next_state] = 0
        predicated_value = reward + self.__discounting*self.__table[next_state]
        self.__table[state] += self.__learning_rate*(predicated_value-self.__table[state])


def discount(r, discounting):
    discounted_reward = np.zeros_like(r, dtype=np.float32)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * discounting + r[i]
        discounted_reward[i] = G
    return discounted_reward

def normalize(scalars):
    mean, std = np.mean(scalars), np.std(scalars)
    return (scalars - mean)/(std+1e-8)

def generalized_adv(rewards, values, discounting, lam=0.95):
    #deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    #self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
    values[-1] = rewards[-1]
    deltas = rewards[:-1] + discounting * values[1:] - values[:-1]
    return np.concatenate([discount(deltas, discounting*lam), [0]], axis=0)