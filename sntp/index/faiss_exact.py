# -*- coding: utf-8 -*-

import faiss

import numpy as np

from sntp.index.base import BaseIndexManager

from typing import Any

try:
    from faiss.swigfaiss import GpuResources
except ImportError:
    GpuResources = Any


class FAISSExactIndexManager(BaseIndexManager):
    def __init__(self,
                 is_cpu: bool = True,
                 resources: GpuResources = None,
                 metric: str = 'l2'):
        super().__init__()
        self.resources = resources
        self.is_cpu = is_cpu
        self.metric = metric

    def query(self,
              index: Any,
              data: np.ndarray,
              k: int = 10) -> np.ndarray:
        data = data.astype(np.float32)
        _, neighbour_indices = index.search(data, k)
        return neighbour_indices

    def create(self,
               data: np.ndarray) -> Any:
        data = data.astype(np.float32)
        dimensionality = data.shape[1]

        if self.metric in {'l2'}:
            cpu_index = faiss.IndexFlatL2(dimensionality)
        else:
            cpu_index = faiss.IndexFlatIP(dimensionality)

        index = cpu_index if self.is_cpu else faiss.index_cpu_to_gpu(self.resources, 0, cpu_index)
        index.add(data)
        return index
