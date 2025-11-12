import lazy_dispatch as ld
import numpy as np
from enum import Enum
from imputer.imputer import Imputer

class ImputeMode(Enum):
    STATIC = "static"
    MEAN = "mean"
    MEDIAN = "median"

class BaselineImputer(Imputer):
    def __init__(self, reference_data, mode=ImputeMode.STATIC, model=None) -> None:
        super().__init__(model=model)
        self.reference_data = reference_data
        self.mode = mode
        print(mode)

    def impute(self, data: object, coalitions: object, mode=ImputeMode.STATIC) -> object:
        if mode == ImputeMode.STATIC:
            return self.impute_static(data, self.reference_data, coalitions)
        elif mode == ImputeMode.MEAN:
            return self.impute_mean(data, self.reference_data, coalitions)
        elif mode == ImputeMode.MEDIAN:
            return self.impute_median(data, self.reference_data, coalitions)
        else:
            raise NotImplementedError(f"Imputation mode {mode} not implemented.")

    @ld.lazydispatch
    def impute_static(data: object, reference_data: object, coalitions: object) -> object:
        return None

    @ld.lazydispatch
    def impute_mean(data: object, reference_data: object, coalitions: object) -> object:
        return None

    @ld.lazydispatch
    def impute_median(data: object, reference_data: object, coalitions: object) -> object:
        return None


    # Implementation for numpy arrays
    @impute_static.register(np.ndarray)
    def impute_static_np(data: np.ndarray, reference_data: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, data.shape[0]))
        
        for coalition_idx in range(num_coalitions):
            imputed = data.copy()
            for i in range(data.shape[0]):
                if coalitions[coalition_idx, i]:
                    imputed[i] = reference_data[i]
            results[coalition_idx] = imputed
        return results if num_coalitions > 1 else results[0]


    @impute_mean.register(np.ndarray)
    def impute_mean_np(data: np.ndarray, reference_data: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, data.shape[0]))
        mean_value = np.mean(reference_data, axis=1)
        
        for coalition_idx in range(num_coalitions):
            imputed = data.copy()
            for i in range(data.shape[0]):
                if coalitions[coalition_idx, i]:
                    imputed[i] = mean_value[i]
            results[coalition_idx] = imputed
        
        return results if num_coalitions > 1 else results[0]


    @impute_median.register(np.ndarray)
    def impute_median_np(data: np.ndarray, reference_data: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, data.shape[0]))
        median_value = np.median(reference_data, axis=1)
        
        for coalition_idx in range(num_coalitions):
            imputed = data.copy()
            for i in range(data.shape[0]):
                if coalitions[coalition_idx, i]:
                    imputed[i] = median_value[i]
            results[coalition_idx] = imputed
        
        return results if num_coalitions > 1 else results[0]


    # Implementation for JAX arrays
    @impute_static.register("jax.Array")
    def impute_static_jax(data: "jax.Array", reference_data: "jax.Array", coalitions: "jax.Array") -> "jax.Array":
        imputed = data.copy()
        for i in range(data.shape[0]):
            if coalitions[i]:
                imputed = imputed.at[i].set(reference_data[i])
        return imputed

    @impute_mean.register("jax.Array")
    def impute_mean_jax(data: "jax.Array", reference_data: "jax.Array", coalitions: "jax.Array") -> "jax.Array":
        from jax import numpy as jnp
        imputed = data.copy()
        mean_value = jnp.mean(reference_data, axis=1).astype(data.dtype)
        for i in range(data.shape[0]):
            if coalitions[i]:
                imputed = imputed.at[i].set(mean_value[i])
        return imputed

    @impute_median.register("jax.Array")
    def impute_median_jax(data: "jax.Array", reference_data: "jax.Array", coalitions: "jax.Array") -> "jax.Array":
        from jax import numpy as jnp
        imputed = data.copy()
        median_value = jnp.median(reference_data, axis=1).astype(data.dtype)
        for i in range(data.shape[0]):
            if coalitions[i]:
                imputed = imputed.at[i].set(median_value[i])
        return imputed


    # Implementation for Polars DataFrames
    @impute_static.register("polars.dataframe.frame.DataFrame")
    def impute_static_polars(data: "polars.dataframe.frame.DataFrame", reference_data: "polars.dataframe.frame.DataFrame", coalitions: "polars.dataframe.frame.DataFrame") -> "polars.dataframe.frame.DataFrame":
        imputed = data.clone()
        for column in data.columns:
            if coalitions[0, column]:
                imputed[0, column] = reference_data[0, column]
        return imputed

    @impute_mean.register("polars.dataframe.frame.DataFrame")
    def impute_mean_polars(data: "polars.dataframe.frame.DataFrame", reference_data: "polars.dataframe.frame.DataFrame", coalitions: "polars.dataframe.frame.DataFrame") -> "polars.dataframe.frame.DataFrame":
        imputed = data.clone()
        mean = reference_data.mean_horizontal()[0]
        for column in data.columns:
            if coalitions[0, column]:
                imputed[0, column] = mean
        return imputed

    @impute_median.register("polars.dataframe.frame.DataFrame")
    def impute_median_polars(data: "polars.dataframe.frame.DataFrame", reference_data: "polars.dataframe.frame.DataFrame", coalitions: "polars.dataframe.frame.DataFrame") -> "polars.dataframe.frame.DataFrame":
        imputed = data.clone()
        median = np.median(reference_data.rows())
        for column in data.columns:
            if coalitions[0, column]:
                imputed[0, column] = median
        return imputed

    # if __name__ == "__main__":
    #     # Test data
    #     data = np.array([4.0, 5.0, 6.0, 7.0])
    #     reference = np.array([9.0, 8.0, 7.0, 6.0])
    #     reference_matrix = np.array([
    #         [10.0, 1.0, 5.0, 3.0],
    #         [20.0, 2.0, 5.0, 6.0],
    #         [30.0, 9.0, 5.0, 9.0],
    #         [40.0, 6.0, 5.0, 12.0]
    #     ])
    #     coalitions = np.array([[0, 1, 0, 1], [1,0,1,0]])
        
    #     print("Original data:", data)
    #     print("reference_data:", reference)
    #     print("coalitions (mask):", coalitions)
    #     print()
        
    #     # Test STATIC mode
    #     print("STATIC mode:")
    #     result_static = impute(data, reference, coalitions, mode=ImputeMode.STATIC)
    #     print(f"Result: {result_static}")
    #     print()
        
    #     # Test MEAN mode with matrix
    #     print("MEAN mode (with reference_matrix):")
    #     result_mean = impute(data, reference_matrix, coalitions, mode=ImputeMode.MEAN)
    #     print(f"Result: {result_mean}")
    #     print()
        
    #     # Test MEDIAN mode with matrix
    #     print("MEDIAN mode (with reference_matrix):")
    #     result_median = impute(data, reference_matrix, coalitions, mode=ImputeMode.MEDIAN)
    #     print(f"Result: {result_median}")

    #     import jax.numpy as jnp
    #     data_jax = jnp.array([4.0, 5.0, 6.0, 7.0])
    #     reference_jax = jnp.array([9.0, 8.0, 7.0, 6.0])
    #     reference_matrix_jax = jnp.array([
    #         [10.0, 1.0, 5.0, 3.0],
    #         [20.0, 2.0, 5.0, 6.0],
    #         [30.0, 9.0, 5.0, 9.0],
    #         [40.0, 6.0, 5.0, 12.0]
    #     ])
    #     coalitions_jax = jnp.array([0, 1, 0, 1])
    #     print(coalitions_jax.size)
    #     print(impute(data_jax, reference_jax, coalitions_jax, mode=ImputeMode.STATIC))

    #     # Test Mean
    #     reference_matrix_jax = jnp.array([
    #         [10, 1, 5, 3],
    #         [20, 2, 5, 6],
    #         [30, 9, 5, 9],
    #         [40, 6, 5, 12]
    #     ])
    #     print(impute(data_jax, reference_matrix_jax, coalitions_jax, mode=ImputeMode.MEAN))

    #     # Test Median
    #     print(impute(data_jax, reference_matrix_jax, coalitions_jax, mode=ImputeMode.MEDIAN))

    #     import polars as pl
    #     data_pl = pl.DataFrame(np.array([[4,5,6,7]]))
    #     reference_pl = pl.DataFrame(np.array([[9,8,7,6]]))
    #     coalitions_pl = pl.DataFrame(np.array([[0,1,0,1]]))
    #     print(type(data_pl))
    #     print(impute(data_pl, reference_pl, coalitions_pl, mode=ImputeMode.STATIC))
