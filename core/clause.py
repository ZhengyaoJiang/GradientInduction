from __future__ import print_function, division, absolute_import
import numpy as np

class Atom(object):
    def __init__(self, predicate, terms):
        '''
        :param predicate: string, the predicate
        :param terms: tuple of size 1 or 2
        '''
        object.__init__(self)
        self.predicate = predicate
        self.terms = terms

    def __hash__(self):
        hashed_list = self.terms[:]
        hashed_list.append(self.predicate)
        return hash(tuple(hashed_list))

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        return not self.__eq__(other)

    def __str__(self):
        terms_str = ""
        for term in self.terms:
            terms_str += str(term)
            terms_str += ","
        terms_str = terms_str[:-1]
        return self.predicate+"("+terms_str+")"


class Clause():
    def __init__(self, head, body):
        '''
        :param head: atom, result of a clause
        :param body: list of atoms, conditions, amximum length is 2
        '''
        self.head = head
        self.body = body

