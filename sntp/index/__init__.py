# -*- coding: utf-8 -*-

from sntp.index.base import BaseIndexManager

from sntp.index.faiss_exact import FAISSExactIndexManager
from sntp.index.faiss_approximate import FAISSApproximateIndexManager

__all__ = [
    'BaseIndexManager',
    'FAISSExactIndexManager',
    'FAISSApproximateIndexManager'
]
