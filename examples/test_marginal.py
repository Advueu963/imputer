import numpy as np
from imputer.marginal_imputer import MarginalImputer, ImputeMode

point = np.array([4.0, 5.0, 6.0, 7.0])
reference_matrix = np.array([
    [10.0, 1.0, 5.0, 3.0],
    [20.0, 2.0, 5.0, 6.0],
    [30.0, 9.0, 5.0, 9.0],
    [40.0, 6.0, 5.0, 12.0]
])
coalitions = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])

imputer_joint = MarginalImputer(reference_matrix, mode=ImputeMode.JOINT, sample_size=10, random_state=42)
result_joint = imputer_joint.impute(point, coalitions)

imputer_indep = MarginalImputer(reference_matrix, mode=ImputeMode.INDEPENDENT, sample_size=10, random_state=42)
result_indep = imputer_indep.impute(point, coalitions)
print("Joint Imputation Result:\n", result_joint)
print("Independent Imputation Result:\n", result_indep)