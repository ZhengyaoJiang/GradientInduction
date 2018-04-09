from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from core.rules import RulesManager

tf.enable_eager_execution()

class Agent(object):
    def __init__(self, rules_manager, background):
        self.rules_manager = rules_manager
        self.labels = None
        self.rule_weights = tf.get_variable()
        self.ground_atoms = rules_manager.all_grounds()
        self.base_valuation = self.background2valuation(background)

    def background2valuation(self, background):
        '''
        :param background: list of Atoms, background knowledge
        :return: a valuation vector
        '''
        return None

    def deduction(self):
        # takes background as input and return a valuation of target ground atoms
        pass

    def inference_step(self):
        pass

    def loss(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(self.deduction(), self.labels)


