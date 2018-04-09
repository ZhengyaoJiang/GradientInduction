from __future__ import print_function, division, absolute_import
import numpy as np
from itertools import product
from core.ilp import *
from core.clause import *
from collections import defaultdict

class RulesManager():
    def __init__(self, language_frame, program_template):
        self.__language = language_frame
        self.__template = program_template

        self.__predicate_mapping = {} # map from predicate to ground atom indices
        self.all_grounds = []
        self.__generate_grounds()

    def generate_clauses(self, intensional, rule_template):
        base_variable = tuple(range(intensional.arity))
        head = (Atom(intensional,base_variable),)

        body_variable = tuple(range(intensional.arity+rule_template.variables_n))
        if rule_template.allow_intensional:
            predicates = list(set(self.__template.auxiliary).union((self.__language.extensional)).union(set([intensional])))
        else:
            predicates = [self.__language.extensional]
        terms = []
        for predicate in predicates:
            body_variables = [body_variable for _ in range(predicate.arity)]
            terms += self.generate_body_atoms(predicate, *body_variables)
        result_tuples = product(head, terms, terms)
        return self.prune([Clause(result[0], result[1:]) for result in result_tuples])

    def find_index(self,atom):
        '''
        find index for a ground atom
        :param atom:
        :return:
        '''
        assert isinstance(atom.terms[0], str)
        assert isinstance(atom.terms[1], str)
        all_indexes = self.__predicate_mapping[atom.predicate]
        for index in all_indexes:
            if self.all_grounds[index] == atom:
                return index
        raise ValueError("didn't find {} in all ground atoms".format(atom))

    def find_satisfy_by_head(self, clause, head):
        result = [] #list of paris of indexes
        free_body = clause.replace_by_head(head).body
        free_variables = list(free_body[0].variables.union(free_body[1].variables))
        repeat_constatns = [self.__language.constants for _ in free_variables]
        all_constants_combination = product(*repeat_constatns)
        all_match = []
        for combination in all_constants_combination:
            all_match.append({free_variables[i]:constant for i,constant in enumerate(combination)})
        for match in all_match:
            result.append((self.find_index(free_body[0].replace_variable(match)),
                           self.find_index(free_body[1].replace_variable(match))))
        return result

    def __generate_grounds(self):
        self.all_grounds.append(Atom(Predicate("Empty", 0), []))
        self.__predicate_mapping[Predicate("Empty", 0)] = [0]
        all_predicates = self.__language.extensional+[self.__language.target]+self.__template.auxiliary
        for predicate in all_predicates:
            constant = self.__language.constants
            constants = [constant for _ in range(predicate.arity)]
            grounds = self.generate_body_atoms(predicate, *constants)
            start = len(self.all_grounds)
            self.all_grounds += grounds
            end = len(self.all_grounds)
            self.__predicate_mapping[predicate] = list(range(start, end))

    @staticmethod
    def prune(clauses):
        pruned = []
        def not_unsafe(clause):
            head_variables = set(clause.head.terms)
            body_variables = set(clause.body[0].terms+clause.body[1].terms)
            return head_variables.issubset(body_variables)

        def not_circular(clause):
            return clause.head not in clause.body

        def not_duplicated(clause):
            for pruned_caluse in pruned:
                if reversed(pruned_caluse.body) == clause.body:
                    return False
            return True

        for clause in clauses:
            if not_unsafe(clause) and not_circular(clause) and not_duplicated(clause):
                pruned.append(clause)
        return pruned


    @staticmethod
    def generate_body_atoms(predicate, *variables):
        '''
        :param predict_candidate: string, candiate of predicate
        :param variables: iterable of tuples of integers, candidates of variables at each position
        :return: tuple of atoms
        '''
        result_tuples = product((predicate,), *variables)
        atoms = [Atom(result[0], result[1:]) for result in result_tuples]
        return atoms

if __name__ == "__main__":
    constants = [str(i) for i in range(10)]
    background = [Atom(Predicate("succ", 2), [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(Predicate("zero", 1), "0"))
    positive = [Atom(Predicate("predecessor", 2), [constants[i + 1], constants[i]]) for i in range(9)]

    all_atom = [Atom(Predicate("predecessor", 2), [constants[i], constants[j]]) for i in range(9) for j in range(9)]
    negative = list(set(all_atom) - set(positive))

    language = LanguageFrame(Predicate("predecessor", 2), [Predicate("zero", 1), Predicate("succ", 2)], constants)
    ilp = ILP(language, background, positive, negative)

    program_temp = ProgramTemplate([], None, None)
    man = RulesManager(language, program_temp)

    clauses = man.generate_clauses(Predicate("predecessor", 2), RuleTemplate(1, True))

    man.find_satisfy_by_head(clauses[0], Atom(Predicate("predecessor", 2), ["1", "2"]))