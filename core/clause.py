from __future__ import print_function, division, absolute_import
import numpy as np

class Atom():
    def __init__(self, predicate, terms):
        '''
        :param predicate: string, the predicate
        :param terms: tuple of size 1 or 2
        '''
        self.predicate = predicate
        self.terms = terms

    def __hash__(self):
        hashed_list = self.terms[:].append(self.predicate)
        return hash(hashed_list)


    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        return not self.__eq__(other)


class Clause():
    def __init__(self, head, body):
        '''
        :param head: atom, result of a clause
        :param body: list of atoms, conditions, amximum length is 2
        '''
        self.head = head
        self.body = body

