# -*- coding: utf-8 -*-

import logging

from abc import ABC

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self):
        pass

    def predict(self, *args, **kwargs):
        raise NotImplementedError
