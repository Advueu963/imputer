import numpy as np
    
def impute_mean(data, coalitions):
    baselineValues = np.mean(data.transpose(), axis=1)
    n_coalitions = coalitions.shape[0]
    ret = np.array([])
    for i in range(n_coalitions):
        for a in range(len(data)):
            datapoint = data[a].copy()
            for b in range(len(datapoint)):
                if coalitions[i][b]:
                    datapoint[b] = baselineValues[a]
            if a == 0:
                temp = datapoint
            else:
                temp = np.stack((temp, datapoint))
        if i == 0:
            ret = temp
        else:
            ret = np.stack((ret, temp))

    return ret

