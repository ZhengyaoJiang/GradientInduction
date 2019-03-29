# -*- coding: utf-8 -*-

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseKernel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x, y):
        raise NotImplementedError
