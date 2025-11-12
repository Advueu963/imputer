import numpy as np
import jax.numpy as jnp
from imputer.baseline_imputer import BaselineImputer, ImputeMode

class DummyModel:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"DummyModel(name='{self.name}')"

    def predict(self, data):
        print(f"[{self.name} predict]: Predicting on data: {data}")
        return data


# Setup
default_model = DummyModel(name="DefaultModel")

# Test data
data = np.array([[4.0, 5.0, 6.0, 7.0]])
reference = np.array([9.0, 8.0, 7.0, 6.0])
reference_matrix = np.array([
    [10.0, 1.0, 5.0, 3.0],
    [20.0, 2.0, 5.0, 6.0],
    [30.0, 9.0, 5.0, 9.0],
    [40.0, 6.0, 5.0, 12.0]
])
coalitions = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])

# Create imputer
static_imputer = BaselineImputer(
    reference_data=reference,
    mode=ImputeMode.STATIC,
    model=default_model
)
mean_imputer = BaselineImputer(
    reference_data=reference_matrix,
    mode=ImputeMode.MEAN,
    model=default_model
)
median_imputer = BaselineImputer(
    reference_data=reference_matrix,
    mode=ImputeMode.MEDIAN,
    model=default_model
)

print("Original data:", data)
print("reference_data:", reference)
print("coalitions (mask):\n", coalitions)

# Test STATIC mode
print("STATIC mode:")
result_static = static_imputer(data, coalitions)
print(f"Result: {result_static}\n")

# Test MEAN mode with matrix
print("MEAN mode (with reference_matrix):")
result_mean = mean_imputer(data, coalitions)
print(f"Result: {result_mean}\n")

# Test MEDIAN mode with matrix
print("MEDIAN mode (with reference_matrix):")
result_median = median_imputer(data, coalitions)
print(f"Result: {result_median}")

import jax.numpy as jnp
data_jax = jnp.array([[4.0, 5.0, 6.0, 7.0]])
reference_jax = jnp.array([9.0, 8.0, 7.0, 6.0])
reference_matrix_jax = jnp.array([
    [10.0, 1.0, 5.0, 3.0],
    [20.0, 2.0, 5.0, 6.0],
    [30.0, 9.0, 5.0, 9.0],
    [40.0, 6.0, 5.0, 12.0]
])
coalitions_jax = jnp.array([0, 1, 0, 1])

static_imputer_jax = BaselineImputer(
    reference_data=reference_jax,
    mode=ImputeMode.STATIC,
    model=default_model
)

result_static_imputer_jax = static_imputer_jax(data_jax, coalitions_jax)
print(f"STATIC mode (JAX): {result_static_imputer_jax}")

# Test Mean
reference_matrix_jax = jnp.array([
    [10, 1, 5, 3],
    [20, 2, 5, 6],
    [30, 9, 5, 9],
    [40, 6, 5, 12]
])

mean_imputer_jax = BaselineImputer(
    reference_data=reference_matrix_jax,
    mode=ImputeMode.MEAN,
    model=default_model
)

result_mean_imputer_jax = mean_imputer_jax(data_jax, coalitions_jax)
print(f"MEAN mode (JAX): {result_mean_imputer_jax}")

# Test Median
median_imputer_jax = BaselineImputer(
    reference_data=reference_matrix_jax,
    mode=ImputeMode.MEDIAN,
    model=default_model
)

result_median_imputer_jax = median_imputer_jax(data_jax, coalitions_jax)
print(f"MEDIAN mode (JAX): {result_median_imputer_jax}")

import polars as pl
data_pl = [pl.DataFrame(np.array([[4,5,6,7]]))]
reference_pl = pl.DataFrame(np.array([[9,8,7,6]]))
coalitions_pl = pl.DataFrame(np.array([[0,1,0,1]]))

static_imputer_pl = BaselineImputer(
    reference_data=reference_pl,
    mode=ImputeMode.STATIC,
    model=default_model
)

result_static_imputer_pl = static_imputer_pl(data_pl, coalitions_pl)
print(f"STATIC mode (Polars): {result_static_imputer_pl}")
