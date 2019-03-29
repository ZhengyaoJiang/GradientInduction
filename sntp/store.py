# -*- coding: utf-8 -*-

import tensorflow as tf

from sntp.util import is_tensor
from sntp.index import BaseIndexManager

from typing import List, Union

import logging

logger = logging.getLogger(__name__)


class Store:
    def __init__(self,
                 index_manager: BaseIndexManager):
        self.index_manager = index_manager
        self.index_store = {}
        self.refresh_index = {}

    @staticmethod
    def _to_key(ae, ge):
        return 'T' if is_tensor(ae) and is_tensor(ge) else 'V'

    # atoms is e.g. [KE, KE, KE], goals is e.g. [GE, X, GE]
    def get(self,
            atoms: List[Union[tf.Tensor, str]],
            goals: List[Union[tf.Tensor, str]],
            position: int) -> object:
        # First create an hash key corresponding to the structure of facts and goals
        # In this case it is TVT
        key = '{}-{}'.format(position, ''.join([self._to_key(ae, ge) for ae, ge in zip(atoms, goals)]))

        if key not in self.index_store or self.refresh_index[key] is True:
            logger.info('Generating index with key {} ..'.format(key))

            ground_atoms = [ae for ae, ge in zip(atoms, goals) if is_tensor(ae) and is_tensor(ge)]
            atom_2d = tf.concat(ground_atoms, axis=1)

            atom_2d_np = atom_2d.numpy()
            index = self.index_manager.create(atom_2d_np)

            self.index_store[key] = index
            self.refresh_index[key] = False

        return self.index_store[key]

    def refresh(self):
        for key in self.index_store:
            self.refresh_index[key] = True
        return
