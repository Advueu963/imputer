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
        return f"PREDICTION_FROM_{self.name.upper()}"


# Setup
default_model = DummyModel(name="DefaultModel")
override_model = DummyModel(name="OverrideModel")

point = np.array([1, 2, 3, 4])
reference = np.array([10, 20, 30, 40])
coalition_mask = np.array([False, True, False, True])

# Create imputer
mean_imputer = BaselineImputer(
    reference_data=reference,
    mode=ImputeMode.MEAN,
    model=default_model
)

# Test with NumPy
result_1 = mean_imputer(data=point, coalitions=coalition_mask)
print(f"NumPy result: {result_1}")

# Test with JAX
point_jax = jnp.array([1, 2, 3, 4])
coalition_mask_jax = jnp.array([False, True, False, True])
result_1b = mean_imputer(data=point_jax, coalitions=coalition_mask_jax)
print(f"JAX result: {result_1b}")