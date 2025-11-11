import lazy_dispatch as ld
import numpy as np
from enum import Enum

class ImputeMode(Enum):
    STATIC = "static"
    MEAN = "mean"
    MEDIAN = "median"

def impute(point : object, reference : object, colations : object, mode=ImputeMode.STATIC) -> object:
    if mode == ImputeMode.STATIC:
        return impute_static(point, reference, colations)
    else:
        raise NotImplementedError(f"Imputation mode {mode} not implemented.")

@ld.lazydispatch
def impute_static(point : object, reference : object, colations : object) -> object:
    return None

@impute_static.register(np.ndarray)
def impute_static_np(point: np.ndarray, reference: np.ndarray, colations: np.ndarray) -> np.ndarray:
    imputed = point.copy()
    for i in range(point.shape[0]):
        if colations[i]:
            imputed[i] = reference[i]
    return imputed

@impute_static.register("jax.Array")
def impute_static_jax(point: "jax.Array", reference: "jax.Array", colations: "jax.Array") -> "jax.Array":
    imputed = point.copy()
    for i in range(point.shape[0]):
        if colations[i]:
            imputed = imputed.at[i].set(reference[i])
    return imputed

if __name__ == "__main__":
    point = np.ndarray(shape=(4,), dtype=int)
    point[:] = [4, 5, 6, 7]
    reference = np.ndarray(shape=(4,), dtype=int)
    reference[:] = [9, 8, 7, 6]
    colations = np.ndarray(shape=(4,), dtype=int)
    colations[:] = [0, 1, 0, 1]
    print(colations.size)
    print(impute(point, reference, colations, mode=ImputeMode.STATIC))

    import jax.numpy as jnp
    point_jax = jnp.array([4, 5, 6, 7])
    reference_jax = jnp.array([9, 8, 7, 6])
    colations_jax = jnp.array([0, 1, 0, 1])
    print(colations_jax.size)
    print(impute(point_jax, reference_jax, colations_jax, mode=ImputeMode.STATIC))
