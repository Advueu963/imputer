import lazy_dispatch as ld
import numpy as np
from enum import Enum
from typing import Optional

class MarginalMode(Enum):
    JOINT = "joint"  # Sample rows jointly (preserves correlations)
    INDEPENDENT = "independent"  # Shuffle columns independently

@ld.lazydispatch
def impute_marginal(
    point: object,
    reference: object,
    colations: object,
    mode: MarginalMode = MarginalMode.JOINT,
    sample_size: int = 100,
    random_state: Optional[int] = None
) -> object:
    """Marginal imputation fallback."""
    return None

@impute_marginal.register(np.ndarray)
def impute_marginal_np(
    point: np.ndarray,
    reference: np.ndarray,
    colations: np.ndarray,
    mode: MarginalMode = MarginalMode.JOINT,
    sample_size: int = 100,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Marginal imputation for numpy arrays.
    
    Args:
        point: Data point to impute (1D array)
        reference: Background data (2D array, rows=samples, cols=features)
        colations: Boolean mask (1=impute, 0=keep original)
        mode: JOINT (sample rows) or INDEPENDENT (shuffle columns)
        sample_size: Number of samples to draw
        random_state: Random seed for reproducibility
        
    Returns:
        Imputed point averaged over samples
    """
    rng = np.random.default_rng(random_state)
    n_samples = min(sample_size, reference.shape[0])
    
    # Prepare replacement data
    if mode == MarginalMode.JOINT:
        sample_indices = rng.choice(reference.shape[0], size=n_samples, replace=False)
        replacement_data = reference[sample_indices]
    else:
        replacement_data = np.copy(reference)
        for col in range(replacement_data.shape[1]):
            rng.shuffle(replacement_data[:, col])
        sample_indices = rng.choice(replacement_data.shape[0], size=n_samples, replace=False)
        replacement_data = replacement_data[sample_indices]
    
    # Impute samples
    imputed_samples = np.zeros((n_samples, point.shape[0]))
    for i in range(n_samples):
        imputed = point.copy()
        for j in range(point.shape[0]):
            if colations[j]:
                imputed[j] = replacement_data[i, j]
        imputed_samples[i] = imputed
    
    # Return mean across samples
    return np.mean(imputed_samples, axis=0)


if __name__ == "__main__":
    # Test marginal imputation
    point = np.array([4.0, 5.0, 6.0, 7.0])
    reference_matrix = np.array([
        [10.0, 1.0, 5.0, 3.0],
        [20.0, 2.0, 5.0, 6.0],
        [30.0, 9.0, 5.0, 9.0],
        [40.0, 6.0, 5.0, 12.0]
    ])
    colations = np.array([0, 1, 0, 1])  # Impute indices 1 and 3
    
    print("="*50)
    print("MARGINAL IMPUTATION TESTS")
    print("="*50)
    print("\nOriginal point:", point)
    print("Reference matrix shape:", reference_matrix.shape)
    print("Colations (mask):", colations)
    print()
    
    # Test JOINT mode
    print("JOINT mode (preserves correlations):")
    result_joint = impute_marginal(
        point, reference_matrix, colations,
        mode=MarginalMode.JOINT,
        sample_size=10,
        random_state=42
    )
    print(f"Result: {result_joint}")
    print()
    
    # Test INDEPENDENT mode
    print("INDEPENDENT mode (breaks correlations):")
    result_indep = impute_marginal(
        point, reference_matrix, colations,
        mode=MarginalMode.INDEPENDENT,
        sample_size=10,
        random_state=42
    )
    print(f"Result: {result_indep}")