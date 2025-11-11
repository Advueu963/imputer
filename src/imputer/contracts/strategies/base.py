from typing import Protocol, Optional
from ..backends import BackendI
from ...helpers.types import BoolMask, ArrayLike, FillValue

class StrategyI(Protocol):
    def fit(self, X:ArrayLike, backend: BackendI, fill_value:  Optional[FillValue] = None):
        raise NotImplementedError
    
    def transform(self, x: ArrayLike, mask: Optional[BoolMask], backend: BackendI):
        raise NotImplementedError
    