import lazy_dispatch as ld
import numpy as np
from enum import Enum
from typing import Optional
from imputer.imputer import Imputer

class MarginalMode(Enum):
    JOINT = "joint"
    INDEPENDENT = "independent"

class MarginalImputer(Imputer):
    def __init__(self, reference_data, mode=MarginalMode.JOINT, sample_size=100, random_state=None, model=None) -> None:
        super().__init__(model=model)
        self.reference_data = reference_data
        self.mode = mode
        self.sample_size = sample_size
        self.random_state = random_state

    def impute(self, data: object, coalitions: object, sample_size: Optional[int] = None, random_state: Optional[int] = None) -> object:
        sample_size = sample_size or self.sample_size
        random_state = random_state if random_state is not None else self.random_state
        
        if self.mode == MarginalMode.JOINT:
            return self.impute_joint(data, self.reference_data, coalitions, sample_size, random_state)
        elif self.mode == MarginalMode.INDEPENDENT:
            return self.impute_independent(data, self.reference_data, coalitions, sample_size, random_state)
        else:
            raise NotImplementedError(f"Imputation mode {self.mode} not implemented.")

    @ld.lazydispatch
    def impute_joint(data: object, reference_data: object, coalitions: object, 
                     sample_size: int, random_state: Optional[int]) -> object:
        return None

    @ld.lazydispatch
    def impute_independent(data: object, reference_data: object, coalitions: object,
                          sample_size: int, random_state: Optional[int]) -> object:
        return None

    # Implementation for numpy arrays - JOINT mode
    @impute_joint.register(np.ndarray)
    def impute_joint_np(data: np.ndarray, reference_data: np.ndarray, coalitions: np.ndarray,
                        sample_size: int, random_state: Optional[int] = None) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, data.shape[0]))
        
        rng = np.random.default_rng(random_state)
        n_samples = min(sample_size, reference_data.shape[0])
        
        for coalition_idx in range(num_coalitions):
            sample_indices = rng.choice(reference_data.shape[0], size=n_samples, replace=False)
            replacement_data = reference_data[sample_indices]
            
            imputed_samples = np.zeros((n_samples, data.shape[0]))
            for i in range(n_samples):
                imputed = data.copy()
                for j in range(data.shape[0]):
                    if coalitions[coalition_idx, j]:
                        imputed[j] = replacement_data[i, j]
                imputed_samples[i] = imputed
            
            results[coalition_idx] = np.mean(imputed_samples, axis=0)
        
        return results if num_coalitions > 1 else results[0]

    # Implementation for numpy arrays - INDEPENDENT mode
    @impute_independent.register(np.ndarray)
    def impute_independent_np(data: np.ndarray, reference_data: np.ndarray, coalitions: np.ndarray,
                             sample_size: int, random_state: Optional[int] = None) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        
        num_coalitions = coalitions.shape[0]
        results = np.zeros((num_coalitions, data.shape[0]))
        
        rng = np.random.default_rng(random_state)
        n_samples = min(sample_size, reference_data.shape[0])
        
        for coalition_idx in range(num_coalitions):
            replacement_data = np.copy(reference_data)
            for col in range(replacement_data.shape[1]):
                rng.shuffle(replacement_data[:, col])
            
            sample_indices = rng.choice(replacement_data.shape[0], size=n_samples, replace=False)
            replacement_data = replacement_data[sample_indices]
            
            imputed_samples = np.zeros((n_samples, data.shape[0]))
            for i in range(n_samples):
                imputed = data.copy()
                for j in range(data.shape[0]):
                    if coalitions[coalition_idx, j]:
                        imputed[j] = replacement_data[i, j]
                imputed_samples[i] = imputed
            
            results[coalition_idx] = np.mean(imputed_samples, axis=0)
        
        return results if num_coalitions > 1 else results[0]

    # Implementation for JAX arrays - JOINT mode
    @impute_joint.register("jax.Array")
    def impute_joint_jax(data: "jax.Array", reference_data: "jax.Array", coalitions: "jax.Array",
                         sample_size: int, random_state: Optional[int] = None) -> "jax.Array":
        from jax import numpy as jnp
        import jax
        
        key = jax.random.PRNGKey(random_state if random_state is not None else 0)
        n_samples = min(sample_size, reference_data.shape[0])
        
        sample_indices = jax.random.choice(key, reference_data.shape[0], shape=(n_samples,), replace=False)
        replacement_data = reference_data[sample_indices]
        
        imputed_samples = []
        for i in range(n_samples):
            imputed = data.copy()
            for j in range(data.shape[0]):
                if coalitions[j]:
                    imputed = imputed.at[j].set(replacement_data[i, j])
            imputed_samples.append(imputed)
        
        return jnp.mean(jnp.stack(imputed_samples), axis=0)

    # Implementation for JAX arrays - INDEPENDENT mode
    @impute_independent.register("jax.Array")
    def impute_independent_jax(data: "jax.Array", reference_data: "jax.Array", coalitions: "jax.Array",
                               sample_size: int, random_state: Optional[int] = None) -> "jax.Array":
        from jax import numpy as jnp
        import jax
        
        key = jax.random.PRNGKey(random_state if random_state is not None else 0)
        n_samples = min(sample_size, reference_data.shape[0])
        
        replacement_data = jnp.array(reference_data)
        for col in range(replacement_data.shape[1]):
            key, subkey = jax.random.split(key)
            replacement_data = replacement_data.at[:, col].set(
                jax.random.permutation(subkey, replacement_data[:, col])
            )
        
        sample_indices = jax.random.choice(key, replacement_data.shape[0], shape=(n_samples,), replace=False)
        replacement_data = replacement_data[sample_indices]
        
        imputed_samples = []
        for i in range(n_samples):
            imputed = data.copy()
            for j in range(data.shape[0]):
                if coalitions[j]:
                    imputed = imputed.at[j].set(replacement_data[i, j])
            imputed_samples.append(imputed)
        
        return jnp.mean(jnp.stack(imputed_samples), axis=0)


if __name__ == "__main__":
    point = np.array([4.0, 5.0, 6.0, 7.0])
    reference_matrix = np.array([
        [10.0, 1.0, 5.0, 3.0],
        [20.0, 2.0, 5.0, 6.0],
        [30.0, 9.0, 5.0, 9.0],
        [40.0, 6.0, 5.0, 12.0]
    ])
    coalitions = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
    
    imputer_joint = MarginalImputer(reference_matrix, mode=MarginalMode.JOINT, sample_size=10, random_state=42)
    result_joint = imputer_joint.impute(point, coalitions)
    
    imputer_indep = MarginalImputer(reference_matrix, mode=MarginalMode.INDEPENDENT, sample_size=10, random_state=42)
    result_indep = imputer_indep.impute(point, coalitions)
    print("Joint Imputation Result:\n", result_joint)
    print("Independent Imputation Result:\n", result_indep)