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
        print(predicates)
        for predicate in predicates:
            print(predicate)
            body_variables = [body_variable for _ in range(predicate.arity)]
            terms += self.generate_term_atoms(predicate, *body_variables)
        result_tuples = product(head, terms, terms)
        return [Clause(result[0], result[1:]) for result in result_tuples]

    @staticmethod
    def generate_term_atoms(predicate, *variables):
        '''
        :param predict_candidate: string, candiate of predicate
        :param variables: iterable of tuples of integers, candidates of variables at each position
        :return: tuple of atoms
        '''
        result_tuples = product((predicate,), *variables)
        atoms = [Atom(result[0], result[1:]) for result in result_tuples]
        return atoms
