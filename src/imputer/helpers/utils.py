from ..contracts.backends import BackendI, NumpyBackend, TorchBackend, JaxBackend
from ..contracts.strategies import StrategyI, MeanStrategy, MedianStrategy, ConstantStrategy
from .enums import BaselineStrategy
from .types import FillValue
from typing import Optional

def pick_backend(x) -> BackendI:
    m = x.__class__.__module__
    if m.startswith("torch"):
        return TorchBackend()
    if m.startswith("jax"):
        return JaxBackend()
    return NumpyBackend()

def pick_baseline_strategy(strategy: BaselineStrategy, fill_value: Optional[FillValue]) -> StrategyI:
    if strategy is BaselineStrategy.median:
        return MedianStrategy()
    if strategy is BaselineStrategy.constant:
        if fill_value is None:
            raise ValueError("'fill_value' was not provided.")
        return ConstantStrategy()
    return MeanStrategy()

def ensure_backend(self, x):
    return self.backend or pick_backend(x)
    
def to_backend(self, backend: BackendI, x):
    return backend.asarray(x, dtype=self.dtype, device=self.device)
    
def to_bool_mask(self, backend: BackendI, xb, mask):
    if mask is None:
        return backend.isnan(xb)
    mb = backend.asarray(mask, dtype=None, device=self.device)
    return mb