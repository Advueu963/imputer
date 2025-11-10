"""Implementation of the baseline imputer."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from .base import Imputer


class BaselineImputer(Imputer):
    def __init__(
        self,
        model: callable,
        data: np.ndarray,
        x: np.ndarray | None = None,
        *,
        categorical_features: list[int] | None = None,
        strategy: Literal["mean", "median", "constant"] = "mean",
        constant_value: float | np.ndarray | None = None,
        normalize: bool = True,
        random_state: int | None = None,
    ) -> None:
        if strategy not in ["mean", "median", "constant"]:
            msg = f"Strategy must be one of 'mean', 'median', or 'constant', got '{strategy}'."
            raise ValueError(msg)

        if strategy == "constant" and constant_value is None:
            msg = "constant_value must be provided when strategy is 'constant'."
            raise ValueError(msg)

        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=1,
            categorical_features=categorical_features,
            random_state=random_state,
        )

        # setup attributes
        self.strategy = strategy
        self.constant_value = constant_value
        self.normalize = normalize
        self.baseline_values: np.ndarray = np.zeros((1, self.n_features))  # will be overwritten
        self.init_background(self.data)

        # set empty value and normalization
        if normalize:
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        n_coalitions = coalitions.shape[0]
        data = np.tile(np.copy(self.x), (n_coalitions, 1))
        for i in range(n_coalitions):
            data[i, ~coalitions[i]] = self.baseline_values[0, ~coalitions[i]]
        return self.predict(data)

    def init_background(self, data: np.ndarray) -> BaselineImputer:
        if data.ndim == 1 or data.shape[0] == 1:  # data is a vector -> use as baseline values
            self.baseline_values = data.reshape(1, self.n_features)
            self.calc_empty_prediction()
            return self

        # data is a matrix -> calculate baseline values based on strategy
        if self.strategy == "constant":
            # Use constant value for all features
            if isinstance(self.constant_value, (int, float)):
                self.baseline_values = np.full((1, self.n_features), self.constant_value, dtype=object)
            else:
                # constant_value is an array
                const_arr = np.asarray(self.constant_value)
                if const_arr.shape[0] != self.n_features:
                    msg = f"constant_value array must have {self.n_features} elements, got {const_arr.shape[0]}."
                    raise ValueError(msg)
                self.baseline_values = const_arr.reshape(1, self.n_features)
        else:
            # Calculate baseline values as mean/median or mode
            self.baseline_values = np.zeros((1, self.n_features), dtype=object)
            for feature in range(self.n_features):
                feature_column = data[:, feature]
                if feature in self._cat_features:  # get mode for categorical features
                    values, counts = np.unique(feature_column, return_counts=True)
                    summarized_feature = values[np.argmax(counts)]
                else:
                    try:  # try to use mean/median for numerical features
                        if self.strategy == "mean":
                            summarized_feature = np.mean(feature_column)
                        elif self.strategy == "median":
                            summarized_feature = np.median(feature_column)
                        else:
                            msg = f"Unknown strategy: {self.strategy}"
                            raise ValueError(msg)
                    except TypeError:  # fallback to mode for potentially string features
                        values, counts = np.unique(feature_column, return_counts=True)
                        summarized_feature = values[np.argmax(counts)]
                        # add feature to categorical features
                        warnings.warn(
                            f"Feature {feature} is not numerical. Adding it to categorical features.",
                            stacklevel=2,
                        )
                        self._cat_features.append(feature)
                self.baseline_values[0, feature] = summarized_feature

        self.calc_empty_prediction()  # reset the empty prediction to the new baseline values
        return self

    def calc_empty_prediction(self) -> float:
        empty_predictions = self.predict(self.baseline_values)
        empty_prediction = float(empty_predictions[0])
        self.empty_prediction = empty_prediction
        if self.normalize:  # reset the normalization value
            self.normalization_value = empty_prediction
        return empty_prediction
