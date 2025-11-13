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

    def impute(self, data: object, coalitions: object) -> object:
        for data_point in data:
            if self.mode == ImputeMode.STATIC:
                yield self.impute_static(data_point, self.reference_data, coalitions)
            elif self.mode == ImputeMode.MEAN:
                yield self.impute_mean(data_point, self.reference_data, coalitions)
            elif self.mode == ImputeMode.MEDIAN:
                yield self.impute_median(data_point, self.reference_data, coalitions)
            else:
                raise NotImplementedError(f"Imputation mode {self.mode} not implemented.")

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
        if coalitions.shape == data.shape:
            coalitions = np.expand_dims(coalitions, axis=0)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, *data.shape), dtype=data.dtype)
        
        for coalition_idx in range(num_coalitions):
            imputed = data.copy()
            imputed[coalitions[coalition_idx]] = reference_data[coalitions[coalition_idx]]
            results[coalition_idx] = imputed
        
        return results if num_coalitions > 1 else results[0]


    @impute_mean.register(np.ndarray)
    def impute_mean_np(data: np.ndarray, reference_data: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.shape == data.shape:
            coalitions = np.expand_dims(coalitions, axis=0)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, *data.shape), dtype=data.dtype)
        
        if reference_data.ndim > data.ndim:
            mean_value = np.mean(reference_data, axis=0)
        else:
            mean_value = reference_data
        
        for coalition_idx in range(num_coalitions):
            imputed = data.copy()
            imputed[coalitions[coalition_idx]] = mean_value[coalitions[coalition_idx]]
            results[coalition_idx] = imputed
        
        return results if num_coalitions > 1 else results[0]

    @impute_median.register(np.ndarray)
    def impute_median_np(data: np.ndarray, reference_data: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.shape == data.shape:
            coalitions = np.expand_dims(coalitions, axis=0)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, *data.shape), dtype=data.dtype)
        
        if reference_data.ndim > data.ndim:
            median_value = np.median(reference_data, axis=0)
        else:
            median_value = reference_data
        
        for coalition_idx in range(num_coalitions):
            imputed = data.copy()
            imputed[coalitions[coalition_idx]] = median_value[coalitions[coalition_idx]]
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
