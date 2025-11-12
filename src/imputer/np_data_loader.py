import numpy as np

def np_data_loader(data, imputation):
    for datapoint in data:
        yield imputation(datapoint)
