from __future__ import print_function, division, absolute_import
import numpy as np
from collections import namedtuple

Predicate = namedtuple("Predicate", "name arity")

class Atom(object):
    def __init__(self, predicate, terms):
        '''
        :param predicate: Predicate, the predicate of the atom
        :param terms: tuple of string (or integer) of size 1 or 2.
        use integer 0, 1, 2 as variables
        '''
        object.__init__(self)
        self.predicate = predicate
        self.terms = tuple(terms)
        assert len(terms)==predicate.arity

    @property
    def arity(self):
        return len(self.terms)

    def __hash__(self):
        hashed_list = list(self.terms[:])
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
        return self.predicate.name+"("+terms_str+")"

    @property
    def variables(self):
        var = []
        for term in self.terms:
            if isinstance(term, int):
                var.append(term)
        return set(var)

    def match_variable(self, target):
        '''
        :param target: ground atom to be matched
        :return: dictionary from int to string, indicating the map from variable to constant. return empty dictionary if
        the two cannot match.
        '''
        assert self.predicate == target.predicate
        match = {}
        for i in range(self.arity):
            if isinstance(self.terms[i], str):
                if self.terms[i] == target.terms[i]:
                    continue
                else:
                    return {}
            else:
                match[self.terms[i]] = target.terms[i]
        return match

    def replace_variable(self, match):
        '''
        :param match: match dictionary
        :return: a atoms whose variable is replaced by constants, given the match mapping.
        '''
        terms = []
        for i,variable in enumerate(self.terms):
            if variable not in match:
                terms.append(variable)
            else:
                terms.append(match[variable])
        result = Atom(self.predicate, terms)
        return result


class Clause():
    def __init__(self, head, body):
        '''
        :param head: atom, result of a clause
        :param body: list of atoms, conditions, amximum length is 2.
        '''
        self.head = head
        self.body = body

    def __str__(self):
        body_str = ""
        for term in self.body:
            body_str += str(term)
            body_str += ","
        body_str = body_str[:-1]
        return str(self.head)+":-"+body_str

    def replace_by_head(self, head):
        '''
        :param head: a ground atom
        :return: replaced clause
        '''
        match = self.head.match_variable(head)
        new_body = []
        for atom in self.body:
            new_body.append(atom.replace_variable(match))
        return Clause(head, new_body)

    def __hash__(self):
        hashed_list = list(self.body[:])
        hashed_list.append(self.head)
        return hash(tuple(hashed_list))

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        return not self.__eq__(other)

