"""Example usage of the BaselineImputer."""

import numpy as np

from imputer import BaselineImputer


def main():
    """Demonstrate the usage of BaselineImputer with different strategies."""
    print("=" * 60)
    print("BaselineImputer Example Usage")
    print("=" * 60)

    # Create a simple model (sum of features)
    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    # Create sample background data
    data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
    ])

    # Point to explain
    x = np.array([2.5, 3.5, 4.5, 5.5])

    print("\nBackground data:")
    print(data)
    print(f"\nPoint to explain: {x}")
    print(f"Model prediction for x: {model(x.reshape(1, -1))[0]}")

    # Example 1: Mean strategy
    print("\n" + "=" * 60)
    print("Example 1: Mean Strategy")
    print("=" * 60)
    imputer_mean = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="mean",
        random_state=42,
    )
    print(f"Baseline values (mean): {imputer_mean.baseline_values[0]}")
    print(f"Empty prediction: {imputer_mean.empty_prediction}")

    # Test with coalitions
    coalitions = np.array([
        [False, False, False, False],  # all missing
        [True, False, False, False],   # only first feature
        [True, True, True, True],      # all features
    ])
    predictions_mean = imputer_mean(coalitions)
    print("\nCoalitions and predictions:")
    for i, (coal, pred) in enumerate(zip(coalitions, predictions_mean)):
        print(f"  Coalition {coal}: prediction = {pred:.2f}")

    # Example 2: Median strategy
    print("\n" + "=" * 60)
    print("Example 2: Median Strategy")
    print("=" * 60)
    imputer_median = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="median",
        random_state=42,
    )
    print(f"Baseline values (median): {imputer_median.baseline_values[0]}")
    print(f"Empty prediction: {imputer_median.empty_prediction}")

    # Example 3: Constant strategy (scalar)
    print("\n" + "=" * 60)
    print("Example 3: Constant Strategy (Scalar)")
    print("=" * 60)
    imputer_constant = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="constant",
        constant_value=0.0,
        random_state=42,
    )
    print(f"Baseline values (constant=0): {imputer_constant.baseline_values[0]}")
    print(f"Empty prediction: {imputer_constant.empty_prediction}")

    # Example 4: Constant strategy (array)
    print("\n" + "=" * 60)
    print("Example 4: Constant Strategy (Array)")
    print("=" * 60)
    constant_values = np.array([1.0, 2.0, 3.0, 4.0])
    imputer_constant_array = BaselineImputer(
        model=model,
        data=data,
        x=x,
        strategy="constant",
        constant_value=constant_values,
        random_state=42,
    )
    print(f"Baseline values (constant array): {imputer_constant_array.baseline_values[0]}")
    print(f"Empty prediction: {imputer_constant_array.empty_prediction}")

    # Example 5: With categorical features
    print("\n" + "=" * 60)
    print("Example 5: With Categorical Features")
    print("=" * 60)
    data_cat = np.array([
        ["red", 1.0, 2.0],
        ["blue", 2.0, 3.0],
        ["red", 3.0, 4.0],
        ["green", 4.0, 5.0],
    ], dtype=object)
    x_cat = np.array(["red", 2.5, 3.5], dtype=object)

    def model_cat(x: np.ndarray) -> np.ndarray:
        # Simple model for mixed data
        return np.array([float(row[1]) + float(row[2]) for row in x])

    imputer_cat = BaselineImputer(
        model=model_cat,
        data=data_cat,
        x=x_cat,
        categorical_features=[0],
        strategy="mean",
        random_state=42,
    )
    print(f"Baseline values (with categorical): {imputer_cat.baseline_values[0]}")
    print(f"  - Feature 0 (categorical): {imputer_cat.baseline_values[0, 0]} (mode)")
    print(f"  - Feature 1 (numerical): {imputer_cat.baseline_values[0, 1]} (mean)")
    print(f"  - Feature 2 (numerical): {imputer_cat.baseline_values[0, 2]} (mean)")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
