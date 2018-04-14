from __future__ import print_function, division, absolute_import

if __name__ == "__main__":
    from core.clause import *
    from core.ilp import *
    from core.rules import *
    from core.induction import *

    constants = [str(i) for i in range(10)]
    background = [Atom(Predicate("succ", 2), [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(Predicate("zero", 1), "0"))
    positive = [Atom(Predicate("predecessor", 2), [constants[i + 1], constants[i]]) for i in range(9)]
    all_atom = [Atom(Predicate("predecessor", 2), [constants[i], constants[j]]) for i in range(9) for j in range(9)]
    negative = list(set(all_atom) - set(positive))

    language = LanguageFrame(Predicate("predecessor",2), [Predicate("zero",1), Predicate("succ",2)], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([], {Predicate("predecessor", 2): [RuleTemplate(1, True), RuleTemplate(1, True)]},
                                   10)
    man = RulesManager(language, program_temp)

    # clauses = man.generate_clauses(Predicate("predecessor", 2), RuleTemplate(1, True))

    agent = Agent(man, ilp)
    agent.train()

    for atom, value in agent.valuation2atoms(agent.deduction()).items():
        print(str(atom)+": "+str(value))