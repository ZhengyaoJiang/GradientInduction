from core.setup import *
from core.hypertune import run
from collections import OrderedDict
import argparse

def generalized_test(task, name, algo):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if task == "cliffwalking":
        env = CliffWalking
    elif task == "unstack":
        env = Unstack
    elif task == "stack":
        env = Stack
    elif task == "on":
        env = On
    elif task == "tictacteo":
        env = TicTacTeo
    else:
        raise ValueError()
    import tensorflow as tf
    summary = OrderedDict()
    if algo=="DILP":
        for variation in [""]+list(env.all_variations):
            print("==========="+variation+"==============")
            result = start_DILP(task, name, "evaluate", variation)
            pprint(result)
            summary[variation] = str(round(result["mean"], 3)) + "+-" + str(round(result["std"], 3))
            tf.reset_default_graph()
    elif algo=="NN":
        for variation in [""]+list(env.all_NN_variations):
            print("==========="+variation+"==============")
            result = start_NN(task, name, "evaluate", variation)
            summary[variation] = str(round(result["mean"], 3)) + "+-" + str(round(result["std"], 3))
            tf.reset_default_graph()
    for k,v in summary.items():
        print(k+": "+v)

#@ray.remote
def start_DILP(task, name, mode, variation=None):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if task == "predecessor":
        man, ilp = setup_predecessor()
        learner = SupervisedDILP(man, ilp, 0.5)
    elif task == "even":
        man, ilp = setup_even()
        learner = SupervisedDILP(man, ilp, 0.5)
    elif task == "cliffwalking":
        man, env = setup_cliffwalking(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        discounting = 1.0
        critic = TableCritic(discounting=discounting, learning_rate=0.1, involve_steps=True)
        # critic = None
        # critic = NeuralCritic([20], env.state_dim, discounting, learning_rate=0.01,
        #                      state2vector=env.state2vector, involve_steps=True)
        learner = ReinforceLearner(agent, env, 0.1, critic=critic, discounting=discounting,
                                   batched=True, steps=12000, name=name)
    elif task == "unstack":
        man, env = setup_unstack(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        # critic = NeuralCritic([10,10], len(env.state))
        # learner = PPOLearner(agent, env, critic)
        if variation:
            critic = None
        else:
            # critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.01, state2vector=env.state2vector)
            critic = TableCritic(discounting=1.0, learning_rate=0.1, involve_steps=True)
            # critic = None
        learner = ReinforceLearner(agent, env, 0.1, critic=critic,
                                   batched=True, steps=50000, name=name)
    elif task == "stack":
        man, env = setup_stack(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        # critic = NeuralCritic([10,10], len(env.state))
        # learner = PPOLearner(agent, env, critic)
        #critic = TableCritic(discounting=1.0)
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.01, state2vector=env.state2vector)
        learner = ReinforceLearner(agent, env, 0.1, critic=critic,
                                   batched=True, steps=12000, name=name)
    elif task == "on":
        man, env = setup_on(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        # critic = NeuralCritic([10,10], len(env.state))
        # learner = PPOLearner(agent, env, critic)
        # critic = None
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.01, state2vector=env.state2vector)
        learner = ReinforceLearner(agent, env, 0.1, critic=critic,
                                   batched=True, steps=12000, name=name)
    elif task == "tictacteo":
        man, env = setup_tictacteo(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        discounting = 0.9
        #critic = TableCritic(discounting, learning_rate=0.2)
        critic = NeuralCritic([20], env.state_dim, discounting, learning_rate=0.01, state2vector=env.state2vector)
        learner = ReinforceLearner(agent, env, 0.02, critic=critic, discounting=discounting,
                                   steps=120000, name=name)

        #learner = PPOLearner(agent, env, 0.02, critic=critic, discounting=discounting,
        #                           steps=120000, name=name)
    else:
        raise ValueError()
    if mode == "train":
        return learner.train()[-1]
    elif mode == "evaluate":
        return learner.evaluate()
    else:
        raise ValueError()

def start_NN(task, name, mode, variation=None):
    if task == "cliffwalking":
        man, env = setup_cliffwalking(variation)
        agent = NeuralAgent([20,10], env.action_n, env.state_dim)
        # critic = TableCritic(1.0)
        #learner = PPOLearner(agent, env, critic=critic)
        critic = None
        learner = ReinforceLearner(agent, env, 0.01, critic=critic,
                                   steps=120000, name=name)
    elif task == "stack":
        man, env = setup_stack(variation)
        agent = NeuralAgent([20,10], env.action_n, env.state_dim)
        critic = TableCritic(1.0)
        learner = ReinforceLearner(agent, env, 0.01, critic=critic,
                                   steps=120000, name=name)
    elif task == "on":
        man, env = setup_on(variation)
        agent = NeuralAgent([200,100], env.action_n, env.state_dim)
        # critic = TableCritic(1.0)
        critic = None
        learner = ReinforceLearner(agent, env, 0.01, critic=critic,
                                   steps=120000, name=name)
    elif task == "tictacteo":
        man, env = setup_tictacteo(variation)
        agent = NeuralAgent([20,10], env.action_n, env.state_dim)
        critic = None
        learner = ReinforceLearner(agent, env, 0.01, critic=critic, discounting=0.9,
                                   steps=120000, name=name)
    if mode == "train":
        return learner.train()[-1]
    elif mode == "evaluate":
        return learner.evaluate()
    else:
        raise ValueError()

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



from pprint import pprint
if __name__ == "__main__":
    #ray.init()
    #print(ray.get([start_DILP.remote("predecessor", "e"+str(i)) for i in range(12)]))
    #start_NTP("predecessor", "predecessor"+"21")
    #run("on")
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--task')
    parser.add_argument('--algo')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    if args.mode=="generalize":
        generalized_test(args.task, args.name, args.algo)
    else:
        if args.algo == "DILP":
            starter = start_DILP
        elif args.algo == "NN":
            starter = start_NN
        else:
            raise ValueError()
        pprint(starter(args.task, args.name, args.mode))

