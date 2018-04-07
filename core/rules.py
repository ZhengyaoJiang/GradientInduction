from __future__ import print_function, division, absolute_import
import numpy as np
from itertools import product
from .ilp import *
from .clause import *

class RulesManager():
    def __init__(self, language_frame, program_template):
        self.__language = language_frame
        self.__template = program_template

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
