import numpy as np

def base_data_loader(data, imputation):
    for datapoint in data:
        yield imputation(datapoint)

def get_base_data_loader(data, coalitions, reference, imputation):
    return base_data_loader(data, (lambda a: imputation(a, reference, coalitions)))