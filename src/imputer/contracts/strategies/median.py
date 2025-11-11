from ..backends import BackendI
from typing import Optional
from ...helpers.types import BoolMask, ArrayLike, FillValue

class MedianStrategy:
    def __init__(self):
        self.col_median: Optional[ArrayLike] = None
    
    def fit(self, X: ArrayLike, backend: BackendI, fill_value:  Optional[FillValue] = None):
        self.col_median = backend.nanmedian(X, axis = 0)
        return self
    
    def transform(self, x: ArrayLike, mask: Optional[BoolMask], backend: BackendI):
        if mask is None:
            mask = backend.isnan(x)
        return backend.where(mask, self.col_median, x)
    
