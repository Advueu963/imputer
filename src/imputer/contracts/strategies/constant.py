from ..backends import BackendI
from typing import Optional
from ...helpers.types import BoolMask, FillValue, ArrayLike

class ConstantStrategy:
    def __init__(self):
        self.col_constant: Optional[ArrayLike] = None
    
    def fit(self, X: ArrayLike, backend: BackendI, fill_value:  Optional[FillValue] = None):
        if fill_value is None:
            raise ValueError("'fill_value' was not provided.")
        self.col_constant = backend.asarray(fill_value, dtype=backend.dtype(X))
        return self
    
    def transform(self, x: ArrayLike, mask: Optional[BoolMask], backend: BackendI):
        if mask is None:
            mask = backend.isnan(x)
        return backend.where(mask, self.col_constant, x)
    
