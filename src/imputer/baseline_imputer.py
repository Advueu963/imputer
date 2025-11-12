import lazy_dispatch as ld
import numpy as np
from enum import Enum
from imputer.imputer import Imputer

class ImputeMode(Enum):
    STATIC = "static"
    MEAN = "mean"
    MEDIAN = "median"

class BaselineImputer(Imputer):
    def __init__(self, mode=ImputeMode.STATIC) -> None:
        super().__init__()
        self.mode = mode

    def impute(self, point: object, data: object, coalitions: object, mode=ImputeMode.STATIC) -> object:
        if mode == ImputeMode.STATIC:
            return self.impute_static(point, data, coalitions)
        elif mode == ImputeMode.MEAN:
            return self.impute_mean(point, data, coalitions)
        elif mode == ImputeMode.MEDIAN:
            return self.impute_median(point, data, coalitions)
        else:
            raise NotImplementedError(f"Imputation mode {mode} not implemented.")

    @ld.lazydispatch
    def impute_static(point : object, reference : object, coalitions : object) -> object:
        return None

    @ld.lazydispatch
    def impute_mean(point : object, reference : object, coalitions : object) -> object:
        return None

    @ld.lazydispatch
    def impute_median(point : object, reference : object, coalitions : object) -> object:
        return None

    # Implementations for numpy arrays
    @impute_static.register(np.ndarray)
    def impute_static_np(point: np.ndarray, reference: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, point.shape[0]))
        
        for coalition_idx in range(num_coalitions):
            imputed = point.copy()
            for i in range(point.shape[0]):
                if coalitions[coalition_idx, i]:
                    imputed[i] = reference[i]
            results[coalition_idx] = imputed
        return results if num_coalitions > 1 else results[0]


    @impute_mean.register(np.ndarray)
    def impute_mean_np(point: np.ndarray, reference: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, point.shape[0]))
        mean_value = np.mean(reference, axis=1)
        
        for coalition_idx in range(num_coalitions):
            imputed = point.copy()
            for i in range(point.shape[0]):
                if coalitions[coalition_idx, i]:
                    imputed[i] = mean_value[i]
            results[coalition_idx] = imputed
        
        return results if num_coalitions > 1 else results[0]


    @impute_median.register(np.ndarray)
    def impute_median_np(point: np.ndarray, reference: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, point.shape[0]))
        median_value = np.median(reference, axis=1)
        
        for coalition_idx in range(num_coalitions):
            imputed = point.copy()
            for i in range(point.shape[0]):
                if coalitions[coalition_idx, i]:
                    imputed[i] = median_value[i]
            results[coalition_idx] = imputed
        
        return results if num_coalitions > 1 else results[0]


    # Implementations for jax arrays
    @impute_static.register("jax.Array")
    def impute_static_jax(point: "jax.Array", reference: "jax.Array", coalitions: "jax.Array") -> "jax.Array":
        imputed = point.copy()
        for i in range(point.shape[0]):
            if coalitions[i]:
                imputed = imputed.at[i].set(reference[i])
        return imputed

    @impute_mean.register("jax.Array")
    def impute_mean_jax(point: "jax.Array", reference: "jax.Array", coalitions: "jax.Array") -> "jax.Array":
        from jax import numpy as jnp
        imputed = point.copy()
        mean_value = jnp.mean(reference, axis=1).astype(point.dtype)
        for i in range(point.shape[0]):
            if coalitions[i]:
                imputed = imputed.at[i].set(mean_value[i])
        return imputed

    @impute_median.register("jax.Array")
    def impute_median_jax(point: "jax.Array", reference: "jax.Array", coalitions: "jax.Array") -> "jax.Array":
        from jax import numpy as jnp
        imputed = point.copy()
        median_value = jnp.median(reference, axis=1).astype(point.dtype)
        for i in range(point.shape[0]):
            if coalitions[i]:
                imputed = imputed.at[i].set(median_value[i])
        return imputed


    # Implementations for polars DataFrame
    @impute_static.register("polars.dataframe.frame.DataFrame")
    def impute_static_polars(point: "polars.dataframe.frame.DataFrame", reference: "polars.dataframe.frame.DataFrame", coalitions: "polars.dataframe.frame.DataFrame") -> "polars.dataframe.frame.DataFrame":
        imputed = point.clone()
        for column in point.columns:
            if coalitions[0, column]:
                imputed[0, column] = reference[0, column]
        return imputed

    @impute_mean.register("polars.dataframe.frame.DataFrame")
    def impute_mean_polars(point: "polars.dataframe.frame.DataFrame", reference: "polars.dataframe.frame.DataFrame", coalitions: "polars.dataframe.frame.DataFrame") -> "polars.dataframe.frame.DataFrame":
        imputed = point.clone()
        mean = reference.mean_horizontal()[0]
        for column in point.columns:
            if coalitions[0, column]:
                imputed[0, column] = mean
        return imputed

    @impute_median.register("polars.dataframe.frame.DataFrame")
    def impute_median_polars(point: "polars.dataframe.frame.DataFrame", reference: "polars.dataframe.frame.DataFrame", coalitions: "polars.dataframe.frame.DataFrame") -> "polars.dataframe.frame.DataFrame":
        imputed = point.clone()
        median = np.median(reference.rows())
        for column in point.columns:
            if coalitions[0, column]:
                imputed[0, column] = median
        return imputed

    # if __name__ == "__main__":
    #     # Test data
    #     point = np.array([4.0, 5.0, 6.0, 7.0])
    #     reference = np.array([9.0, 8.0, 7.0, 6.0])
    #     reference_matrix = np.array([
    #         [10.0, 1.0, 5.0, 3.0],
    #         [20.0, 2.0, 5.0, 6.0],
    #         [30.0, 9.0, 5.0, 9.0],
    #         [40.0, 6.0, 5.0, 12.0]
    #     ])
    #     coalitions = np.array([[0, 1, 0, 1], [1,0,1,0]])
        
    #     print("Original point:", point)
    #     print("Reference:", reference)
    #     print("coalitions (mask):", coalitions)
    #     print()
        
    #     # Test STATIC mode
    #     print("STATIC mode:")
    #     result_static = impute(point, reference, coalitions, mode=ImputeMode.STATIC)
    #     print(f"Result: {result_static}")
    #     print()
        
    #     # Test MEAN mode with matrix
    #     print("MEAN mode (with reference_matrix):")
    #     result_mean = impute(point, reference_matrix, coalitions, mode=ImputeMode.MEAN)
    #     print(f"Result: {result_mean}")
    #     print()
        
    #     # Test MEDIAN mode with matrix
    #     print("MEDIAN mode (with reference_matrix):")
    #     result_median = impute(point, reference_matrix, coalitions, mode=ImputeMode.MEDIAN)
    #     print(f"Result: {result_median}")

    #     import jax.numpy as jnp
    #     point_jax = jnp.array([4.0, 5.0, 6.0, 7.0])
    #     reference_jax = jnp.array([9.0, 8.0, 7.0, 6.0])
    #     reference_matrix_jax = jnp.array([
    #         [10.0, 1.0, 5.0, 3.0],
    #         [20.0, 2.0, 5.0, 6.0],
    #         [30.0, 9.0, 5.0, 9.0],
    #         [40.0, 6.0, 5.0, 12.0]
    #     ])
    #     coalitions_jax = jnp.array([0, 1, 0, 1])
    #     print(coalitions_jax.size)
    #     print(impute(point_jax, reference_jax, coalitions_jax, mode=ImputeMode.STATIC))

    #     # Test Mean
    #     reference_matrix_jax = jnp.array([
    #         [10, 1, 5, 3],
    #         [20, 2, 5, 6],
    #         [30, 9, 5, 9],
    #         [40, 6, 5, 12]
    #     ])
    #     print(impute(point_jax, reference_matrix_jax, coalitions_jax, mode=ImputeMode.MEAN))

    #     # Test Median
    #     print(impute(point_jax, reference_matrix_jax, coalitions_jax, mode=ImputeMode.MEDIAN))

    #     import polars as pl
    #     point_pl = pl.DataFrame(np.array([[4,5,6,7]]))
    #     reference_pl = pl.DataFrame(np.array([[9,8,7,6]]))
    #     coalitions_pl = pl.DataFrame(np.array([[0,1,0,1]]))
    #     print(type(point_pl))
    #     print(impute(point_pl, reference_pl, coalitions_pl, mode=ImputeMode.STATIC))
