from core.setup import *

#@ray.remote
def start_DILP(task, name):
    import tensorflow as tf
    if task == "predecessor":
        man, ilp = setup_predecessor()
        learner = SupervisedDILP(man, ilp, 0.5)
    elif task == "even":
        man, ilp = setup_even()
        learner = SupervisedDILP(man, ilp, 0.5)
    elif task == "cliffwalking":
        man, env = setup_cliffwalking()
        agent = RLDILP(man, env)
        # critic = NeuralCritic([10,10], len(env.state))
        # learner = PPOLearner(agent, env, critic)
        learner = ReinforceLearner(agent, env, 0.1)
    elif task == "unstack":
        man, env = setup_unstack()
        agent = RLDILP(man, env, state_encoding="atoms")
        # critic = NeuralCritic([10,10], len(env.state))
        # learner = PPOLearner(agent, env, critic)
        learner = ReinforceLearner(agent, env, 0.02)
    elif task == "stack":
        man, env = setup_stack()
        agent = RLDILP(man, env, state_encoding="atoms")
        # critic = NeuralCritic([10,10], len(env.state))
        # learner = PPOLearner(agent, env, critic)
        critic = TableCritic(1.0)
        learner = ReinforceLearner(agent, env, 0.5, critic=critic)
    elif task == "on":
        man, env = setup_on()
        agent = RLDILP(man, env, state_encoding="atoms")
        # critic = NeuralCritic([10,10], len(env.state))
        # learner = PPOLearner(agent, env, critic)
        # critic = None
        critic = TableCritic(1.0)
        learner = ReinforceLearner(agent, env, 0.1, critic=critic)
    else:
        raise ValueError()
    return learner.train(batched=True, steps=6000, name=name)[-1]

def start_NN(task, name=None):
    if task == "cliffwalking":
        man, env = setup_cliffwalking()
        agent = NeuralAgent([20,10], len(env.actions), len(env.state))
        # critic = NeuralCritic([10,10], len(env.state))
        critic = None
        #learner = PPOLearner(agent, env, critic=critic)
        learner = ReinforceLearner(agent, env)
    return learner.train(steps=6000, name=name, learning_rate=1e-3)[-1]

#@ray.remote
def start_NTP(task, name=None):
    import tensorflow as tf
    from core.NTP import ProofState
    tf.enable_eager_execution()
    if task == "predecessor":
        man, ilp = setup_predecessor()
        ntp = NeuralProver.from_ILP(ilp, [str2clause("predecessor(X,Y):-s1(X,Z),s2(Z,Y)"),
                                                  str2clause("predecessor(X,Y):-s3(X,X),s4(X,Y)"),
                                                  str2clause("predecessor(X,Y):-s5(X,X),s6(Y,Y)"),
                                                  str2clause("predecessor(X,Y):-s7(X,Y),s8(Y,Y)"),
                                                  str2clause("predecessor(X,Y):-s9(Y,X)")
                                                  ])
        final_loss = ntp.train(ilp.positive,ilp.negative,2,3000)[-1]
    if task == "even":
        man, ilp = setup_even()
        ntp = NeuralProver.from_ILP(ilp, [str2clause("predecessor(X,Y):-s(X,Z),s2(Z,Y)"),
                                          str2clause("even(Y):-p(X,Y),e(X)"),
                                          str2clause("even(X):-z(X)")])
        final_loss = ntp.train(ilp.positive,ilp.negative,2,3000)[-1]
    if task == "cliffwalking":
        man, env = setup_cliffwalking()
        agent = RLProver.from_Env(env,
                                  [
                                   str2clause("a1(X,Y):-s1(X)"),
                                   str2clause("a2(X,Y):-s2(Y)"),
                                   str2clause("a3(X,Y):-s3(Z,X)"),
                                   str2clause("a4(X,Y):-s4(Z,Y)"),
                                   str2clause("a5(X,Y):-s5(X,Z)"),
                                   str2clause("a6(X,Y):-s6(Y,Z)"),
                                   str2clause("a7(X,Y):-s7(X,Y),s8(X,Z)"),
                                   str2clause("a8(X,Y):-s9(Y,Z),s10(X,Z)"),
                                   str2clause("a9(X,Y):-s11(Y,X),s12(Z,Y)"),
                                   str2clause("a10(X,Y):-s13(X,Y),s14(Z,Y)"),
                                   ]
                                  ,2)
        learner = ReinforceLearner(agent, env)
        final_loss = learner.train(steps=2000, name=name, learning_rate=0.001, optimizer="Adam")
    return final_loss

if __name__ == "__main__":
    #ray.init()
    #print(ray.get([start_DILP.remote("predecessor", "e"+str(i)) for i in range(12)]))
    #start_NTP("predecessor", "predecessor"+"21")
    with tf.device("cpu"):
        #start_DILP("cliffwalking", "102000")
        start_DILP("on", "on13")
        #start_NTP("cliffwalking", "NTPRL08")
        #start_NTP("predecessor", None)
