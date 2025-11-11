from typing import Optional
from . import BaselineStrategy
from .contracts.backends import BackendI

from .helpers.types import ArrayLike

from .helpers.types import BoolMask,FillValue, ImputationResult
from .helpers.utils import ensure_backend, to_backend, to_bool_mask, pick_backend, pick_baseline_strategy

class BaselineImputer():
    def __init__(
            self, 
            strategy: BaselineStrategy = BaselineStrategy.mean,
            dtype: Optional[str] = None,
            device: Optional[str] = None, 
            fill_value: Optional[FillValue] = None
        ):
        self.backend: Optional[BackendI] = None
        self.strategy = pick_baseline_strategy(strategy, fill_value)
        self.dtype, self.device = dtype, device
        self.fill_value = fill_value

    def fit(self, X: ArrayLike):
        self.backend = pick_backend(X)
        Xb = self.backend.asarray(X, dtype=self.dtype, device=self.device)
        self.strategy.fit(Xb, self.backend, self.fill_value)
        return self
    
    def transform(self, x: ArrayLike, mask: Optional[BoolMask]) -> ImputationResult:
        backend = ensure_backend(self, x)
        x_backend = to_backend(self, backend, x)
        mask_backend = to_bool_mask(self, backend, x_backend, mask)

        return ImputationResult(X=self.strategy.transform(x_backend, mask_backend, backend))