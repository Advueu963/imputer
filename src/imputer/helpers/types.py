from dataclasses import dataclass
from torch import Tensor
from typing import Sequence, Union, Optional
from jax import Array as JaxArray
from numpy import ndarray as NumpyArray


ArrayLike = Union[
    NumpyArray,
    Tensor,
    JaxArray,
    Sequence,  # 1D python list/tuple
    Sequence[Sequence],  # 2D python list/tuple
]

BoolMask = Union[
    NumpyArray,
    Tensor,
    JaxArray,
    Sequence[bool],  # lists/tuples of bools
]

FillValue = Union[
    float,
    ArrayLike
]

@dataclass
class ImputationResult:
    X: Optional[ArrayLike] = None