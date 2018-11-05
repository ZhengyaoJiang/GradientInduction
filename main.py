from __future__ import print_function, division, absolute_import
from core.clause import *
from core.ilp import *
import ray
from core.rules import *
from core.induction import *
from core.clause import str2atom,str2clause
from core.NTP import NeuralProver, RLProver, SymbolicNeuralProver
from core.symbolicEnvironment import *

def setup_predecessor():
    constants = [str(i) for i in range(10)]
    background = [Atom(Predicate("succ", 2), [constants[i], constants[i + 1]]) for i in range(9)]
    positive = [Atom(Predicate("predecessor", 2), [constants[i], constants[i+2]]) for i in range(8)]
    all_atom = [Atom(Predicate("predecessor", 2), [constants[i], constants[j]]) for i in range(10) for j in range(10)]
    negative = list(set(all_atom) - set(positive))

    language = LanguageFrame(Predicate("predecessor",2), [Predicate("succ",2)], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([], {Predicate("predecessor", 2): [RuleTemplate(1, False), RuleTemplate(0, False)]},
                                   4)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_fizz():
    constants = [str(i) for i in range(10)]
    succ = Predicate("succ", 2)
    zero = Predicate("zero", 1)
    fizz = Predicate("fizz", 1)
    pred1 = Predicate("pred1", 2)
    pred2 = Predicate("pred2", 2)

    background = [Atom(succ, [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(zero, "0"))
    positive = [Atom(fizz, [constants[i]]) for i in range(0, 10, 3)]
    all_atom = [Atom(fizz, [constants[i]]) for i in range(10)]
    negative = list(set(all_atom) - set(positive))
    language = LanguageFrame(fizz, [zero, succ], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([pred1, pred2], {fizz: [RuleTemplate(1, True), RuleTemplate(1, False)],
                                                    pred1: [RuleTemplate(1, True),],
                                                    pred2: [RuleTemplate(1, True),],},
                                   10)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_even():
    constants = [str(i) for i in range(10)]
    succ = Predicate("succ", 2)
    zero = Predicate("zero", 1)
    target = Predicate("even", 1)
    pred = Predicate("pred", 2)
    background = [Atom(succ, [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(zero, "0"))
    positive = [Atom(target, [constants[i]]) for i in range(0, 10, 2)]
    all_atom = [Atom(target, [constants[i]]) for i in range(10)]
    negative = list(set(all_atom) - set(positive))
    language = LanguageFrame(target, [zero, succ], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([pred], {target: [RuleTemplate(1, True), RuleTemplate(1, False)],
                                            pred: [RuleTemplate(1, True),RuleTemplate(1, False)],
                                            },
                                   10)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_cliffwalking(invented=False):
    env = CliffWalking()
    if invented:
        temp = [RuleTemplate(1, False), RuleTemplate(1, True)]
        invented = Predicate("invented", 2)
        program_temp = ProgramTemplate([invented], {invented:temp, UP: temp, DOWN: temp, LEFT: temp, RIGHT: temp}, 2)
    else:
        temp = [RuleTemplate(2, False)]
        program_temp = ProgramTemplate([], {UP: temp, DOWN: temp, LEFT: temp, RIGHT: temp}, 1)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_unstack():
    env = Unstack()
    maintemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    inventedtemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    invented = Predicate("invented", 2)
    program_temp = ProgramTemplate([invented], {invented:inventedtemp, MOVE:maintemp}, 3)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_stack():
    env = Stack(initial_state=INI_STATE2)
    maintemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    inventedtemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    invented = Predicate("invented", 2)
    program_temp = ProgramTemplate([invented], {invented:inventedtemp, MOVE:maintemp}, 3)
    man = RulesManager(env.language, program_temp)
    return man, env




#@ray.remote
def start_DILP(task, name):
    import tensorflow as tf
    if task == "predecessor":
        man, ilp = setup_predecessor()
        learner = SupervisedDILP(man, ilp)
        learning_rate = 0.5
    elif task == "even":
        man, ilp = setup_even()
        learner = SupervisedDILP(man, ilp)
        learning_rate = 0.5
    elif task == "cliffwalking":
        man, env = setup_cliffwalking()
        agent = RLDILP(man, env)
        # critic = NeuralCritic([10,10], len(env.state))
        # learner = PPOLearner(agent, env, critic)
        learner = ReinforceLearner(agent, env)
        learning_rate = 0.1
    elif task == "unstack":
        man, env = setup_unstack()
        agent = RLDILP(man, env, state_encoding="atoms")
        # critic = NeuralCritic([10,10], len(env.state))
        # learner = PPOLearner(agent, env, critic)
        learner = ReinforceLearner(agent, env)
        learning_rate = 0.5
    elif task == "stack":
        man, env = setup_stack()
        agent = RLDILP(man, env, state_encoding="atoms")
        # critic = NeuralCritic([10,10], len(env.state))
        # learner = PPOLearner(agent, env, critic)
        learner = ReinforceLearner(agent, env)
        learning_rate = 0.1
    else:
        raise ValueError()
    return learner.train(steps=6000, name=name, batched=False, learning_rate=learning_rate)[-1]

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
    tf.enable_eager_execution()
    with tf.device("cpu"):
        #start_DILP("cliffwalking", "102000")
        start_DILP("stack", "stack10")
        #start_NTP("cliffwalking", "NTPRL08")
        #start_NTP("predecessor", None)
