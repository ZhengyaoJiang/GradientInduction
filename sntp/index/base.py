# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np

from typing import Any

import logging

logger = logging.getLogger(__name__)


class BaseIndexManager(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def query(self,
              index: Any,
              data: np.ndarray,
              k: int = 10) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def create(self,
               data: np.ndarray) -> Any:
        raise NotImplementedError
