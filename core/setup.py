from __future__ import print_function, division, absolute_import
from core.clause import *
from core.ilp import *
import ray
from core.rules import *
from core.induction import *
from core.rl import *
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
    maintemp = [RuleTemplate(1, True)]
    inventedtemp = [RuleTemplate(1, True)]
    invented = Predicate("invented", 2)
    invented2 = Predicate("invented2", 1)
    program_temp = ProgramTemplate([invented, invented2], {invented:inventedtemp, MOVE:maintemp,
                                                           invented2:inventedtemp}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_stack():
    env = Stack(initial_state=INI_STATE2)
    maintemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    inventedtemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    invented = Predicate("invented", 2)
    invented2 = Predicate("invented2", 2)
    program_temp = ProgramTemplate([invented, invented2],
                                   {invented:inventedtemp, MOVE:maintemp, invented2:inventedtemp}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_on():
    env = On()
    maintemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    inventedtemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    invented = Predicate("invented", 2)
    invented2 = Predicate("invented2", 2)
    program_temp = ProgramTemplate([invented, invented2], {invented:inventedtemp, MOVE:maintemp,
                                                           invented2:inventedtemp}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_tictacteo():
    env = TicTacTeo()
    maintemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    inventedtemp = [RuleTemplate(1, False), RuleTemplate(1, True)]
    invented = Predicate("invented", 2)
    invented2 = Predicate("invented2", 2)
    program_temp = ProgramTemplate([invented, invented2], {invented:inventedtemp, PLACE:maintemp,
                                                           invented2:inventedtemp}, 3)
    man = RulesManager(env.language, program_temp)
    return man, env

