from .baseline import BaselineImputer
from .helpers.enums import BaselineStrategy
import torch


def main():
    # Create a torch tensor with NaNs
    X = torch.tensor([
        [1.0, float("nan")],
        [float("nan"), 4.0],
        [3.0, 2.0],
    ], dtype=torch.float32)

    print("Original X:\n", X)

    strategies = [
        ("mean", BaselineStrategy.mean, {}),
        ("median", BaselineStrategy.median, {}),
        ("constant (0)", BaselineStrategy.constant, {"fill_value": 0}),
    ]

    for label, strategy, kwargs in strategies:
        imputer = BaselineImputer(strategy, **kwargs)
        imputer.fit(X)
        res = imputer.transform(X, mask=None)
        print(f"\nImputed ({label}):\n", res.X)

        # Mask as a torch.bool tensor
        mask = torch.tensor([
            [True, False],
            [False, True],
            [False, False],
        ], dtype=torch.bool)

        res_masked = imputer.transform(X, mask=mask)
        print(f"Imputed ({label}) with mask:\n", res_masked.X)


if __name__ == "__main__":
    main()

# from .baseline import BaselineImputer
# from .helpers.enums import BaselineStrategy
# import numpy as np


# def main():
#     X = np.array([
#         [1.0, np.nan],
#         [np.nan, 4.0],
#         [3.0, 2.0],
#     ])

#     print("Original X:\n", X)

#     strategies = [
#         ("mean", BaselineStrategy.mean, {}),
#         ("median", BaselineStrategy.median, {}),
#         ("constant (0)", BaselineStrategy.constant, {"fill_value": 0}),
#     ]

#     for label, strategy, kwargs in strategies:
#         imputer = BaselineImputer(strategy, **kwargs)
#         imputer.fit(X)
#         res = imputer.transform(X, mask=None)
#         print(f"\nImputed ({label}):\n", res.X)
#         mask = np.array([
#             [True, False],
#             [False, True],
#             [False, False],
#         ])
#         res_masked = imputer.transform(X, mask=mask)
#         print(f"Imputed ({label}) with mask:\n", res_masked.X)


# if __name__ == "__main__":
#     main()
