"""Tests for the baseline imputer module."""

from __future__ import annotations
import numpy as np
import pytest
from imputer import BaselineImputer

def test_baseline_imputer_init_with_mean_strategy():
    """Test the initialization of the baseline imputer with mean strategy."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    x = np.array([1.0, 2.0, 3.0])

    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="mean",
        random_state=42,
    )

    # Check that baseline values are computed as mean
    expected_baseline = np.array([[2.0, 3.0, 4.0]])
    assert np.allclose(imputer.baseline_values, expected_baseline)
    assert imputer.strategy == "mean"
    assert imputer.n_features == 3


def test_baseline_imputer_init_with_median_strategy():
    """Test the initialization of the baseline imputer with median strategy."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    x = np.array([1.0, 2.0, 3.0])

    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="median",
        random_state=42,
    )

    # Check that baseline values are computed as median
    expected_baseline = np.array([[2.0, 3.0, 4.0]])
    assert np.allclose(imputer.baseline_values, expected_baseline)
    assert imputer.strategy == "median"


def test_baseline_imputer_init_with_constant_strategy():
    """Test the initialization of the baseline imputer with constant strategy."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    x = np.array([1.0, 2.0, 3.0])

    # Test with scalar constant
    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="constant",
        constant_value=0.0,
        random_state=42,
    )

    expected_baseline = np.array([[0.0, 0.0, 0.0]])
    assert np.allclose(imputer.baseline_values, expected_baseline)
    assert imputer.strategy == "constant"


def test_baseline_imputer_constant_array():
    """Test the baseline imputer with constant values as an array."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    x = np.array([1.0, 2.0, 3.0])

    constant_values = np.array([1.0, 2.0, 3.0])
    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="constant",
        constant_value=constant_values,
        random_state=42,
    )

    expected_baseline = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(imputer.baseline_values, expected_baseline)


def test_baseline_imputer_with_baseline_vector():
    """Test initialization with a baseline vector instead of computing from data."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0])

    baseline_vector = np.array([0.0, 0.0, 0.0])
    x = np.array([1.0, 2.0, 3.0])

    imputer = BaselineImputer(
        model=model,
        data=baseline_vector,
        x=x,
        random_state=42,
    )

    expected_baseline = np.array([[0.0, 0.0, 0.0]])
    assert np.allclose(imputer.baseline_values, expected_baseline)


def test_baseline_imputer_value_function():
    """Test the value function of the baseline imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    x = np.array([1.0, 2.0, 3.0])

    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="mean",
        random_state=42,
        normalize=False,
    )

    # Test with coalitions
    coalitions = np.array([
        [False, False, False],  # all missing -> use all baseline values
        [True, False, False],    # first feature present
        [True, True, True],      # all features present
    ])

    predictions = imputer.value_function(coalitions)

    # For coalition [False, False, False], should use baseline [2.0, 3.0, 4.0]
    # For coalition [True, False, False], should use [1.0, 3.0, 4.0] (x[0]=1, baseline for rest)
    # For coalition [True, True, True], should use [1.0, 2.0, 3.0] (all from x)

    assert predictions.shape == (3,)
    assert np.isclose(predictions[0], 9.0)  # 2 + 3 + 4
    assert np.isclose(predictions[1], 8.0)  # 1 + 3 + 4
    assert np.isclose(predictions[2], 6.0)  # 1 + 2 + 3


def test_baseline_imputer_with_categorical_features():
    """Test the baseline imputer with categorical features."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0])

    # Create data with categorical features
    data = np.array([["a", 1.0, 2.0], ["b", 2.0, 3.0], ["a", 3.0, 4.0]], dtype=object)
    x = np.array(["a", 1.0, 2.0], dtype=object)

    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        categorical_features=[0],  # first feature is categorical
        strategy="mean",
        random_state=42,
    )

    # For categorical feature, should use mode
    # For numerical features, should use mean
    assert imputer.baseline_values[0, 0] == "a"  # mode of ["a", "b", "a"]
    assert np.isclose(imputer.baseline_values[0, 1], 2.0)  # mean of [1, 2, 3]
    assert np.isclose(imputer.baseline_values[0, 2], 3.0)  # mean of [2, 3, 4]


def test_baseline_imputer_init_background():
    """Test the init_background method."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0])

    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    x = np.array([1.0, 2.0, 3.0])

    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="mean",
        random_state=42,
    )

    # Change baseline values
    new_baseline = np.array([0.0, 0.0, 0.0])
    imputer.init_background(new_baseline)

    expected_baseline = np.array([[0.0, 0.0, 0.0]])
    assert np.allclose(imputer.baseline_values, expected_baseline)


def test_baseline_imputer_invalid_strategy():
    """Test that invalid strategy raises an error."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0]])
    x = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Strategy must be one of"):
        BaselineImputer(
            model=model,
            data=data,
            x=x,
            strategy="invalid",  # type: ignore
            random_state=42,
        )


def test_baseline_imputer_constant_without_value():
    """Test that using constant strategy without providing constant_value raises an error."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0]])
    x = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="constant_value must be provided"):
        BaselineImputer(
            model=model,
            data=data,
            x=x,
            strategy="constant",
            random_state=42,
        )


def test_baseline_imputer_constant_wrong_shape():
    """Test that constant_value array with wrong shape raises an error."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])  # multiple rows
    x = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="constant_value array must have"):
        BaselineImputer(
            model=model,
            data=data,
            x=x,
            strategy="constant",
            constant_value=np.array([1.0, 2.0]),  # wrong shape
            random_state=42,
        )



def test_baseline_imputer_fit():
    """Test the fit method of the baseline imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0]])

    # Create imputer without x
    imputer = BaselineImputer(
        model=model,
        data=data,
        x=None,
        strategy="mean",
        random_state=42,
    )

    # Fit with a new x
    x = np.array([1.0, 2.0, 3.0])
    imputer.fit(x)

    assert np.array_equal(imputer.x, x.reshape(1, -1))
    assert imputer.n_features == 3


def test_baseline_imputer_call():
    """Test calling the imputer directly."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    x = np.array([1.0, 2.0, 3.0])

    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="mean",
        random_state=42,
        normalize=False,
    )

    coalitions = np.array([[True, True, False]])
    predictions = imputer(coalitions)

    # Should use [1.0, 2.0, 3.5] (x[0], x[1], baseline[2])
    assert predictions.shape == (1,)
    assert np.isclose(predictions[0], 6.5)


def test_baseline_imputer_median_vs_mean():
    """Test that median and mean strategies produce different results with skewed data."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    # Skewed data where median != mean
    data = np.array([[1.0], [1.0], [1.0], [10.0]])
    x = np.array([5.0])

    imputer_mean = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="mean",
        random_state=42,
    )

    imputer_median = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="median",
        random_state=42,
    )

    # Mean should be (1+1+1+10)/4 = 3.25
    # Median should be 1.0
    assert np.isclose(imputer_mean.baseline_values[0, 0], 3.25)
    assert np.isclose(imputer_median.baseline_values[0, 0], 1.0)


def test_baseline_imputer_empty_prediction():
    """Test that empty prediction is calculated correctly."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([1.0, 2.0])

    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="mean",
        random_state=42,
    )

    # Empty prediction should be sum of baseline values: 2.0 + 3.0 = 5.0
    assert np.isclose(imputer.empty_prediction, 5.0)


def test_baseline_imputer_sample_size():
    """Test that sample size is always 1 for baseline imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.array([[1.0, 2.0, 3.0]])
    x = np.array([1.0, 2.0, 3.0])

    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="mean",
        random_state=42,
    )

    # Sample size should always be 1 for baseline imputer
    assert imputer.sample_size == 1
