from abc import ABC, abstractmethod
from typing import Union, Tuple, Literal
import numpy as np
import pandas as pd
import polars as pl
import torch

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ==================== Backend Layer ====================

class TensorBackend(ABC):
    """Abstract base class for tensor operations"""
    
    @abstractmethod
    def sample_background(self, background, n_samples: int):
        """Sample n_samples rows from background data"""
        pass
    
    @abstractmethod
    def create_mask(self, shape: Tuple[int, ...], mask_prob: float):
        """Create binary mask with given probability"""
        pass
    
    @abstractmethod
    def compute_mean(self, data, axis: int):
        """Compute mean along axis"""
        pass
    
    @abstractmethod
    def compute_median(self, data, axis: int):
        """Compute median along axis"""
        pass
    
    @abstractmethod
    def expand_dims(self, data, axis: int):
        """Add new dimension at axis"""
        pass
    
    @abstractmethod
    def tile(self, data, reps):
        """Tile data according to reps"""
        pass
    
    @abstractmethod
    def zeros_like(self, data):
        """Create zero tensor with same shape and device"""
        pass


class PyTorchBackend(TensorBackend):
    
    def sample_background(self, background, n_samples):
        indices = torch.randint(0, len(background), (n_samples,))
        return background[indices]
    
    def create_mask(self, shape, mask_prob):
        return torch.bernoulli(torch.full(shape, 1 - mask_prob))
    
    def compute_mean(self, data, axis):
        return data.mean(dim=axis, keepdim=True)
    
    def compute_median(self, data, axis):
        return data.median(dim=axis, keepdim=True).values
    
    def expand_dims(self, data, axis):
        return data.unsqueeze(axis)
    
    def tile(self, data, reps):
        return data.repeat(*reps)
    
    def zeros_like(self, data):
        return torch.zeros_like(data)


class JAXBackend(TensorBackend):
    """JAX backend for GPU/TPU acceleration"""
    
    def __init__(self, rng_key=None):
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")
        self.rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
    
    def sample_background(self, background, n_samples):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        indices = jax.random.randint(subkey, (n_samples,), 0, len(background))
        return background[indices]
    
    def create_mask(self, shape, mask_prob):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return jax.random.bernoulli(subkey, 1 - mask_prob, shape).astype(jnp.float32)
    
    def compute_mean(self, data, axis):
        return jnp.mean(data, axis=axis, keepdims=True)
    
    def compute_median(self, data, axis):
        return jnp.median(data, axis=axis, keepdims=True)
    
    def expand_dims(self, data, axis):
        return jnp.expand_dims(data, axis=axis)
    
    def tile(self, data, reps):
        return jnp.tile(data, reps)
    
    def zeros_like(self, data):
        return jnp.zeros_like(data)


class NumpyBackend(TensorBackend):
    """NumPy backend (also handles Pandas/Polars)"""
    
    def sample_background(self, background, n_samples):
        indices = np.random.randint(0, len(background), n_samples)
        return background[indices]
    
    def create_mask(self, shape, mask_prob):
        return (np.random.rand(*shape) > mask_prob).astype(np.float32)
    
    def compute_mean(self, data, axis):
        return np.mean(data, axis=axis, keepdims=True)
    
    def compute_median(self, data, axis):
        return np.median(data, axis=axis, keepdims=True)
    
    def expand_dims(self, data, axis):
        return np.expand_dims(data, axis=axis)
    
    def tile(self, data, reps):
        return np.tile(data, reps)
    
    def zeros_like(self, data):
        return np.zeros_like(data)


# ==================== Strategy Layer ====================

class ImputationStrategy(ABC):
    """Abstract base class for imputation strategies"""
    
    @abstractmethod
    def compute_fill_value(self, background, backend: TensorBackend):
        """Compute the value(s) to use for imputation"""
        pass
    
    def impute(self, X, background, mask, backend: TensorBackend, n_imputations: int):
        """
        Perform imputation using the strategy
        
        Args:
            X: Input data (batch_size, features)
            background: Background dataset (n_background, features)
            mask: Binary mask (batch_size, n_imputations, features)
            backend: TensorBackend instance
            n_imputations: Number of imputations per sample
        
        Returns:
            Imputed data (batch_size, n_imputations, features)
        """
        batch_size, n_features = X.shape[:2]
        
        # Expand X: (batch, features) -> (batch, n_imputations, features)
        X_expanded = backend.expand_dims(X, 1)
        X_expanded = backend.tile(X_expanded, (1, n_imputations, 1))
        
        # Get fill values
        fill_values = self.compute_fill_value(background, backend)
        
        # Broadcast fill_values to match shape
        # fill_values shape could be (features,) or (n_samples, features)
        if len(fill_values.shape) == 1:
            # Expand to (1, 1, features) for broadcasting
            fill_values = backend.expand_dims(fill_values, 0)
            fill_values = backend.expand_dims(fill_values, 0)
        else:
            # Reshape to (batch, n_imputations, features)
            fill_values = fill_values.reshape(batch_size, n_imputations, n_features)
        
        # Apply mask: keep X where mask=1, use fill_values where mask=0
        return X_expanded * mask + fill_values * (1 - mask)


class StaticImputation(ImputationStrategy):
    """Fill masked values with a static value (e.g., 0)"""
    
    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value
    
    def compute_fill_value(self, background, backend: TensorBackend):
        # Return a tensor of zeros (or the static value) with same shape as background features
        dummy = backend.zeros_like(background[0])
        return dummy + self.fill_value


class MeanImputation(ImputationStrategy):
    """Fill masked values with feature-wise mean from background"""
    
    def compute_fill_value(self, background, backend: TensorBackend):
        # Compute mean across samples: (n_background, features) -> (features,)
        mean_values = backend.compute_mean(background, axis=0)
        return mean_values.squeeze()


class MedianImputation(ImputationStrategy):
    """Fill masked values with feature-wise median from background"""
    
    def compute_fill_value(self, background, backend: TensorBackend):
        # Compute median across samples: (n_background, features) -> (features,)
        median_values = backend.compute_median(background, axis=0)
        return median_values.squeeze()


# ==================== Adapter Layer ====================

class DataFrameAdapter:
    """Convert DataFrames to tensors and vice versa"""
    
    def __init__(self, data):
        self.original_type = type(data)
        self.column_names = None
        self.index = None
        
        # Extract metadata for DataFrames
        if isinstance(data, pd.DataFrame):
            self.column_names = data.columns.tolist()
            self.index = data.index
        elif isinstance(data, pl.DataFrame):
            self.column_names = data.columns
        
        # Convert to tensor
        self.tensor, self.backend = self._convert_to_tensor(data)
    
    def _convert_to_tensor(self, data):
        """Convert input to tensor and select appropriate backend"""
        if isinstance(data, torch.Tensor):
            return data, PyTorchBackend()
        
        elif JAX_AVAILABLE and isinstance(data, jnp.ndarray):
            return data, JAXBackend()
        
        elif isinstance(data, np.ndarray):
            return data, NumpyBackend()
        
        elif isinstance(data, pd.DataFrame):
            # Pandas -> NumPy (copy to avoid side effects)
            return data.values.copy(), NumpyBackend()
        
        elif isinstance(data, pl.DataFrame):
            # Polars -> NumPy (already copies)
            return data.to_numpy(), NumpyBackend()
        
        else:
            raise TypeError(f"Unsupported type: {type(data)}")


# ==================== Main API ====================

class Imputer:
    """
    Main imputation API supporting multiple backends and strategies
    
    Supported backends:
        - PyTorch (GPU/CPU)
        - JAX (GPU/TPU/CPU)
        - NumPy (CPU)
        - Pandas (converted to NumPy)
        - Polars (converted to NumPy)
    
    Supported strategies:
        - 'static': Fill with constant value (default 0)
        - 'mean': Fill with feature-wise mean from background
        - 'median': Fill with feature-wise median from background
    """
    
    def __init__(
        self, 
        background_data,
        strategy: Literal['static', 'mean', 'median'] = 'mean',
        static_value: float = 0.0
    ):
        """
        Initialize imputer
        
        Args:
            background_data: Background dataset for computing statistics
            strategy: Imputation strategy ('static', 'mean', or 'median')
            static_value: Value to use for 'static' strategy
        """
        self.adapter = DataFrameAdapter(background_data)
        self.background = self.adapter.tensor
        self.backend = self.adapter.backend
        
        # Select strategy
        if strategy == 'static':
            self.strategy = StaticImputation(fill_value=static_value)
        elif strategy == 'mean':
            self.strategy = MeanImputation()
        elif strategy == 'median':
            self.strategy = MedianImputation()
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from 'static', 'mean', 'median'")
    
    def impute(
        self, 
        X, 
        n_imputations: int = 100,
        mask_prob: float = 0.5,
        return_original_format: bool = False
    ):
        """
        Perform imputation on input data
        
        Args:
            X: Input data to impute (supports torch/numpy/jax/pandas/polars)
            n_imputations: Number of imputations per sample
            mask_prob: Probability of masking each feature (0-1)
            return_original_format: Whether to convert back to original format
        
        Returns:
            Imputed data with shape (batch_size, n_imputations, n_features)
        """
        # Convert input
        X_adapter = DataFrameAdapter(X)
        X_tensor = X_adapter.tensor
        
        # Check backend compatibility
        if type(self.backend) != type(X_adapter.backend):
            raise ValueError(
                f"Background uses {type(self.backend).__name__} but X uses "
                f"{type(X_adapter.backend).__name__}. They must match."
            )
        
        batch_size, n_features = X_tensor.shape
        
        # Generate mask: (batch_size, n_imputations, n_features)
        mask = self.backend.create_mask(
            (batch_size, n_imputations, n_features),
            mask_prob
        )
        
        # Move mask to same device as X (for PyTorch/JAX)
        if isinstance(X_tensor, torch.Tensor):
            mask = mask.to(X_tensor.device)
        
        # Perform imputation using strategy
        result = self.strategy.impute(
            X_tensor, 
            self.background, 
            mask, 
            self.backend,
            n_imputations
        )
        
        # Convert back to original format if requested
        if return_original_format:
            return self._restore_format(result, X_adapter)
        
        return result
    
    def _restore_format(self, result, adapter: DataFrameAdapter):
        """Convert result back to original format (takes first imputation only)"""
        # Extract first imputation: (batch, n_imp, features) -> (batch, features)
        if isinstance(result, torch.Tensor):
            result_2d = result[:, 0, :].cpu().numpy()
        elif JAX_AVAILABLE and isinstance(result, jnp.ndarray):
            result_2d = np.array(result[:, 0, :])
        else:
            result_2d = result[:, 0, :]
        
        if adapter.original_type == pd.DataFrame:
            return pd.DataFrame(
                result_2d,
                columns=adapter.column_names,
                index=adapter.index
            )
        elif adapter.original_type == pl.DataFrame:
            return pl.DataFrame(result_2d, schema=adapter.column_names)
        elif adapter.original_type == torch.Tensor:
            return torch.from_numpy(result_2d)
        elif JAX_AVAILABLE and adapter.original_type == jnp.ndarray:
            return jnp.array(result_2d)
        else:
            return result_2d