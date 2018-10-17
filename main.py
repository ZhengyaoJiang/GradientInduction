from __future__ import print_function, division, absolute_import
from core.clause import *
from core.ilp import *
import ray
from core.rules import *
from core.induction import *
from core.clause import str2atom,str2clause
from core.NTP import NeuralProver
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

def setup_cliffwalking():
    env = CliffWalking()
    temp = [RuleTemplate(1, False)]
    program_temp = ProgramTemplate([], {UP: temp, DOWN: temp, LEFT: temp, RIGHT: temp}, 1)
    man = RulesManager(env.language, program_temp)
    return man, env


#@ray.remote
def start_DILP(task, name):
    import tensorflow as tf
    if task == "predecessor":
        man, ilp = setup_predecessor()
        agent = SupervisedDILP(man, ilp)
    elif task == "even":
        man, ilp = setup_even()
        agent = SupervisedDILP(man, ilp)
    elif task == "cliffwalking":
        man, env = setup_cliffwalking()
        agent = ReinforceDILP(man, env)
    else:
        raise ValueError()
    return agent.train(steps=6000, name=name)[-1]

@ray.remote
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
    if task == "even":
        man, ilp = setup_even()
        ntp = NeuralProver.from_ILP(ilp, [str2clause("predecessor(X,Y):-s(X,Z),s2(Z,Y)"),
                                          str2clause("even(Y):-p(X,Y),e(X)"),
                                          str2clause("even(X):-z(X)")])
    final_loss = ntp.train(ilp.positive,ilp.negative,2,3000)[-1]
    return final_loss

if __name__ == "__main__":
    #ray.init()
    #print(ray.get([start_DILP.remote("predecessor", "e"+str(i)) for i in range(12)]))
    #start_NTP("predecessor", "predecessor"+"21")
    tf.enable_eager_execution()
    with tf.device("cpu"):
        start_DILP("cliffwalking", "cliff5_fix_softmaxbug")
