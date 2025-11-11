from ..backends import BackendI
from typing import Optional
from ...helpers.types import BoolMask, ArrayLike

class MostFrequentStrategy:
    def __init__(self):
        self.col_mode: Optional[ArrayLike] = None

    def fit(self, X: ArrayLike, backend: BackendI):
        raise NotImplementedError
    
    def transform(self, x: ArrayLike, mask: Optional[BoolMask], backend: BackendI):
        if mask is None:
            mask = backend.isnan(x)
        return backend.where(mask, self.col_mode, x)
    
