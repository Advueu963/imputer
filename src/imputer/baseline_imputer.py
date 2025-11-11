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
    elif mode == ImputeMode.MEAN:
        return impute_mean(point, reference, colations)
    elif mode == ImputeMode.MEDIAN:
        return impute_median(point, reference, colations)
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

@ld.lazydispatch
def impute_mean(point : object, reference : object, colations : object) -> object:
    return None

@impute_mean.register(np.ndarray)
def impute_mean_np(point: np.ndarray, reference: np.ndarray, colations: np.ndarray) -> np.ndarray:
    imputed = point.copy()
    mean_value = np.mean(reference, axis=1)
    for i in range(point.shape[0]):
        if colations[i]:
            imputed[i] = mean_value[i]
    return imputed

@ld.lazydispatch
def impute_median(point : object, reference : object, colations : object) -> object:
    return None

@impute_median.register(np.ndarray)
def impute_median_np(point: np.ndarray, reference: np.ndarray, colations: np.ndarray) -> np.ndarray:
    imputed = point.copy()
    median_value = np.median(reference, axis=1)
    for i in range(point.shape[0]):
        if colations[i]:
            imputed[i] = median_value[i]
    return imputed



@impute_static.register("jax.Array")
def impute_static_jax(point: "jax.Array", reference: "jax.Array", colations: "jax.Array") -> "jax.Array":
    imputed = point.copy()
    for i in range(point.shape[0]):
        if colations[i]:
            imputed = imputed.at[i].set(reference[i])
    return imputed

if __name__ == "__main__":
    # Test data    point = np.array([4.0, 5.0, 6.0, 7.0])
    reference = np.array([9.0, 8.0, 7.0, 6.0])
    reference_matrix = np.array([
        [10.0, 1.0, 5.0, 3.0],
        [20.0, 2.0, 5.0, 6.0],
        [30.0, 9.0, 5.0, 9.0],
        [40.0, 6.0, 5.0, 12.0]
    ])
    colations = np.array([0, 1, 0, 1])
    
    print("Original point:", point)
    print("Reference:", reference)
    print("Colations (mask):", colations)
    print()
    
    # Test STATIC mode
    print("STATIC mode:")
    result_static = impute(point, reference, colations, mode=ImputeMode.STATIC)
    print(f"Result: {result_static}")
    print()
    
    # Test MEAN mode with matrix
    print("MEAN mode (with reference_matrix):")
    result_mean = impute(point, reference_matrix, colations, mode=ImputeMode.MEAN)
    print(f"Result: {result_mean}")
    print()
    
    # Test MEDIAN mode with matrix
    print("MEDIAN mode (with reference_matrix):")
    result_median = impute(point, reference_matrix, colations, mode=ImputeMode.MEDIAN)
    print(f"Result: {result_median}")

    import jax.numpy as jnp
    point_jax = jnp.array([4, 5, 6, 7])
    reference_jax = jnp.array([9, 8, 7, 6])
    colations_jax = jnp.array([0, 1, 0, 1])
    print(colations_jax.size)
    print(impute(point_jax, reference_jax, colations_jax, mode=ImputeMode.STATIC))
