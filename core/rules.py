from __future__ import print_function, division, absolute_import
import numpy as np
from itertools import product
from .ilp import *
from .clause import *

class RulesManager():
    def __init__(self, language_frame, program_template):
        self.__language = language_frame
        self.__template = program_template

    def generate_clauses(self, target, rule_template):
        base_variable = tuple(range(target.arity))
        base_variables = [base_variable for _ in target.arity]
        head = self.generate_predicates((target.predicate,), *base_variables)

        term1 = self.generate_predicates(self.__template.auxiliary+self.__language.extension, )


    @staticmethod
    def generate_predicates(predicate_candidate, *variables):
        '''
        :param predict_candidate: string, candiates of predicate
        :param variables: iterable of integer, candidates of variables
        :return: tuple of atoms
        '''
        result_tuples = product((predicate_candidate,), *variables)
        atoms = [Atom(result[0], result[1:]) for result in result_tuples]
        return atoms
