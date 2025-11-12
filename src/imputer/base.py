"""Base class for all Imputers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np


class Imputer(ABC):
    """Base class for Imputers.

    Attributes:
        n_features: The number of features in the data.
        data: The background data to use for the imputer.
        model: The model to impute missing values for as a callable function.
        sample_size: The number of samples to draw from the background data.
        random_state: The random state to use for sampling.
        empty_prediction: The model's prediction on an empty data point (all features missing).

    Properties:
        x: The explanation point to use the imputer on.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        x: np.ndarray | None = None,
        *,
        sample_size: int | None = 100,
        categorical_features: list[int] | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initializes the base imputer.

        Args:
            model: The model to explain as a callable function expecting data points as input and
                returning the model's predictions.
            data: The background data to use for the imputer as a 2-dimensional array with shape
                ``(n_samples, n_features)``.
            x: The explanation point to use the imputer on either as a 2-dimensional array with
                shape ``(1, n_features)`` or as a vector with shape ``(n_features,)``.
            sample_size: The number of samples to draw from the background data. Defaults to ``100``.
            categorical_features: A list of indices of the categorical features in the background data.
            random_state: The random state to use for sampling. Defaults to ``None``.
        """
        if not callable(model):
            msg = "The model must be callable."
            raise ValueError(msg)
        self.model = model

        # check if data is a vector
        if data.ndim == 1:
            data = data.reshape(1, data.shape[0])
        self.data = data

        self.n_features = self.data.shape[1]
        self._sample_size = sample_size
        self.empty_prediction: float = 0.0  # will be overwritten in the subclasses
        self._cat_features: list[int] = [] if categorical_features is None else categorical_features
        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)

        # fit x
        self._x: np.ndarray | None = None
        if x is not None:
            self.fit(x)

    @property
    def x(self) -> np.ndarray:
        """Returns the explanation point if it is set."""
        if self._x is None:
            msg = "The imputer has not yet been fitted yet."
            raise AttributeError(msg)
        return self._x.copy()

    @property
    def sample_size(self) -> int:
        """Returns the sample size."""
        if self._sample_size is None:
            msg = "The sample size is not set."
            raise AttributeError(msg)
        return self._sample_size

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Provides a unified prediction interface.

        Args:
            x: The data point to predict the model's output for.

        Returns:
            The model's prediction for the given data point as a vector.
        """
        return self.model(x)

    def fit(self, x: np.ndarray) -> Imputer:
        """Fits the imputer to the explanation point.

        Args:
            x: The explanation point to use the imputer on either as a 2-dimensional array with
                shape ``(1, n_features)`` or as a vector with shape ``(n_features,)``.

        Returns:
            The fitted imputer.
        """
        self._x = x.copy()
        if self._x.ndim == 1:
            self._x = self._x.reshape(1, x.shape[0])
        return self

    @abstractmethod
    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Imputes the missing values of a data point and calls the model.

        Args:
            coalitions: A boolean array indicating which features are present (``True``) and which are
                missing (``False``). The shape of the array must be ``(n_subsets, n_features)``.

        Returns:
            The model's predictions on the imputed data points.
        """
        raise NotImplementedError

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Calls the value function of the imputer.

        Args:
            coalitions: A boolean array indicating which features are present and which are missing.

        Returns:
            The model's predictions on the imputed data points.
        """
        return self.value_function(coalitions)

    @abstractmethod
    def init_background(self, data: np.ndarray) -> Imputer:
        """Initializes the imputer to the background data.

        Args:
            data: The background data to use for the imputer.

        Returns:
            The initialized imputer.
        """
        raise NotImplementedError
