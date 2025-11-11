from enum import Enum

class BaselineStrategy(Enum):
    mode = 'mode',
    mean = "mean"
    median = "median"
    most_frequent = "most_frequent"
    constant = "constant"