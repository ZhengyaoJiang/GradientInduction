from core.induction import *
from core.NTP import *
from copy import deepcopy

class TableCritic(object):
    def __init__(self, discounting, learning_rate=0.1):
        self.__table = {}
        self.__discounting = discounting
        self.__learning_rate = learning_rate

    def batch_learn(self, states, rewards, next_states):
        for s, a, s2 in zip(states, rewards, next_states):
            self.learn(s, a, s2)

    def get_advantage(self, rewards, states):
        values = np.array([self.__table[tuple(state)] for state in states] + [0.0])
        return rewards - values[:-1] + self.__discounting*values[1:]

    def learn(self, state, reward, next_state):
        state = tuple(state)
        next_state = tuple(next_state)
        if state not in self.__table:
            self.__table[state] = 0
        if next_state not in self.__table:
            self.__table[next_state] = 0
        predicated_value = reward + self.__discounting*self.__table[next_state]
        self.__table[state] += self.__learning_rate*(predicated_value-self.__table[state])


class ReinforceLearner(object):
    def __init__(self, agent, enviornment, learning_rate, critic=None):
        # super(ReinforceDILP, self).__init__(rules_manager, enviornment.background)
        self.env = enviornment
        self.agent = agent
        self.state_encoding = self.agent.state_encoding
        self.learning_rate = learning_rate
        self._construct_train(learning_rate)
        self.critic=critic

    def _construct_train(self, learning_rate):
        self.tf_returns = tf.placeholder(shape=[None], dtype=tf.float32)
        #self.tf_episode_n = tf.placeholder(shape=[])
        self.tf_advantage = tf.placeholder(shape=[None], dtype=tf.float32)
        self.tf_additional_discount = tf.placeholder(shape=[None], dtype=tf.float32)
        self.tf_valuation_index = tf.placeholder(shape=[None, self.env.action_n], dtype=tf.int32)
        self.tf_action_index = tf.placeholder(shape=[None], dtype=tf.int32)
        self._construct_action_prob()
        indexed_action_prob = tf.batch_gather(self.tf_action_prob, self.tf_action_index[:, None])[:, 0]
        self.tf_loss =-tf.reduce_sum(tf.log(tf.clip_by_value(indexed_action_prob, 1e-5, 1.0))\
                      *self.tf_advantage*self.tf_additional_discount)
        #self.tf_loss = tf.Print(self.tf_loss, [self.tf_loss])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        #grads = self.grad()
        #self.tf_grads = grads
        #self.tf_train = self.optimizer.apply_gradients(zip(grads, self.agent.all_variables()),
        #                          global_step=tf.train.get_or_create_global_step())
        self.tf_train = self.optimizer.minimize(self.tf_loss, tf.train.get_or_create_global_step(),
                                                var_list=self.agent.all_variables())

    def _construct_action_prob(self):
        if self.state_encoding == "atoms":
            if isinstance(self.agent, RLDILP):
                action_eval = tf.batch_gather(self.agent.tf_result_valuation , self.tf_valuation_index)
                sum_action_eval = tf.reduce_sum(action_eval)
                action_prob = tf.where(sum_action_eval > 1.0,
                                       action_eval / sum_action_eval,
                                       action_eval + (1.0 - sum_action_eval) / self.env.action_n)
                self.tf_action_prob = action_prob / tf.reduce_sum(action_prob)

    def grad(self):
        loss_value = self.tf_loss
        weight_decay = 0.0
        regularization = 0
        for weights in self.agent.all_variables():
            weights = tf.nn.softmax(weights)
            regularization += tf.reduce_sum(tf.sqrt(weights))*weight_decay
        loss_value += regularization/len(self.agent.all_variables())
        return tf.gradients(loss_value, self.agent.all_variables())

    def sample_episode(self, sess, max_steps=99999):
        action_prob_history = []
        action_history = []
        reward_history = []
        action_trajectory_prob = []
        valuation_history = []
        state_history = []
        excesses = []
        valuation_index_history = []
        step = 0
        while step<max_steps:
            step += 1
            indexes = self.agent.get_valuation_indexes(self.env.state2atoms(self.env.state))
            if self.state_encoding=="terms":
                valuation = self.agent.base_valuation
            else:
                valuation = self.agent.base_valuation + self.agent.axioms2valuation(self.env.state2atoms(self.env.state))
            action_prob,result = sess.run([self.tf_action_prob, self.agent.tf_result_valuation], feed_dict={self.agent.tf_input_valuation: [valuation],
                                                                     self.tf_valuation_index: [indexes]})
            action_prob = action_prob[0]
            action_index = np.random.choice(range(self.env.action_n), p=action_prob)
            if self.state_encoding == "terms":
                action = self.env.action_index2symbol(action_index)
            else:
                action = self.agent.all_actions[action_index]
            state_history.append(self.env.state)
            reward, finished = self.env.next_step(action)
            reward_history.append(reward)
            action_history.append(action_index)
            action_trajectory_prob.append(action_prob[action_index])
            valuation_history.append(valuation)
            valuation_index_history.append(indexes)
            if finished:
                self.env.reset()
                break
        return reward_history, action_history, action_trajectory_prob, state_history, valuation_history, valuation_index_history


    def summary_scalar(self, name, scalar):
        if self.summary_writer:
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar(name, scalar)

    def summary_histogram(self, name, data):
        if self.summary_writer:
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.histogram(name, data)

    def train(self, steps=300, name=None, discounting=1.0, batched=True, optimizer="RMSProp"):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            losses = []
            sess.run([tf.initializers.global_variables()])
            if name:
                try:
                    saver.restore(sess, "./model/"+name+"/parameters.ckpt")
                except Exception as e:
                    print(e)
                self.summary_writer = tf.contrib.summary.create_file_writer("./model/"+name, flush_millis=10000)
            else:
                self.summary_writer = None
            self.summary_scalar("returns", self.tf_returns[0])
            self.summary_histogram("advantages", self.tf_advantage)
            self.summary_scalar("loss", self.tf_loss)
            self.summary_histogram("weights", tf.concat(self.agent.all_variables(), axis=0))
                # model definition code goes here
                # and in it call
            with self.summary_writer.as_default():
                tf.contrib.summary.initialize(graph=tf.get_default_graph(), session=sess)
            for i in range(steps):
                reward_history, action_history, action_prob_history, state_history, valuation_history, valuation_index_history = \
                        self.sample_episode(sess)
                returns = discount(reward_history, discounting)
                if self.critic:
                    self.critic.batch_learn(state_history, reward_history, state_history[1:]+["end"])
                    advnatage = self.critic.get_advantage(reward_history, state_history)
                else:
                    advnatage = returns
                additional_discount = np.cumprod(discounting*np.ones_like(advnatage))
                log = {"return":returns[0], "action_history":[str(self.agent.all_actions[action_index])
                                                                  for action_index in action_history]}

                if batched:
                    _, _ = sess.run([self.tf_train, tf.contrib.summary.all_summary_ops()],
                                        {self.tf_advantage:np.array(advnatage), self.tf_additional_discount:np.array(additional_discount),
                                         self.tf_returns:np.array(returns),
                                         self.tf_action_index:np.array(action_history),
                                         self.tf_valuation_index: np.array(valuation_index_history),
                                         self.agent.tf_input_valuation: np.array(valuation_history)})
                else:
                    first = True
                    for action_index, adv, acc_discount, val, val_index in zip(action_history, advnatage, additional_discount,
                                                                  valuation_history,valuation_index_history):
                        ops = [self.tf_train, self.tf_loss, self.tf_action_prob]
                        if first == True:
                            ops += [tf.contrib.summary.all_summary_ops()]
                            first = False
                        result = sess.run(ops, {self.tf_advantage: [adv],
                                         self.tf_additional_discount: [acc_discount],
                                       self.tf_returns: np.array(returns),
                                       self.tf_action_index: [action_index],
                                       self.tf_valuation_index: [val_index],
                                       self.agent.tf_input_valuation: [val]})
                        #print("advantage: {}".format(adv))
                        #print("action prob: {}".format(result[2][0][action_index]))
                        #print("loss value: {}".format(result[1]))
                print("-"*20)
                print("step "+str(i)+"return is "+str(log["return"]))
                if i%10==0:
                    self.agent.log(sess)
                    pprint(log)
                    if name:
                        saver.save(sess, "./model/" + name + "/parameters.ckpt")
                        pd.Series(np.array(losses)).to_csv(name+".csv")
                print("-"*20+"\n")
        return log["return"]

class PPOLearner(ReinforceLearner):
    def __init__(self, agent, enviornment, learning_rate, critic=None):
        super(PPOLearner, self).__init__(agent, enviornment, learning_rate)
        self.epsilon = 0.1
        self.critic = critic
        self.state_size = len(enviornment.state)
        self.discounting = 1.0

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
        model.add(tf.keras.layers.Dense(unit_list[0], input_shape=(state_size,), activation=tf.nn.sigmoid,
                                        kernel_initializer=tf.initializers.random_normal()))
        for i in range(1, len(unit_list)):
            model.add(tf.keras.layers.Dense(unit_list[i], activation=tf.nn.sigmoid,
                                            kernel_initializer=tf.initializers.random_normal()))
        model.add(tf.keras.layers.Dense(action_size, activation=tf.nn.softmax,
                                        ))
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
        model.add(tf.keras.layers.Dense(1, kernel_initializer=tf.initializers.random_normal()))
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