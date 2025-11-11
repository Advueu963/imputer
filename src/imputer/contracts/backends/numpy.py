from numpy import ndarray

class NumpyBackend:
    import numpy as np

    def isnan(self, x: ndarray): return self.np.isnan(x)

    def nanmean(self, x: ndarray, axis = 0): return self.np.nanmean(x, axis)

    def nanmedian(self, x: ndarray, axis = 0): return self.np.nanmean(x, axis)

    def mode(self, x: ndarray, axis = 0, keep_dim=False): 
        vals, counts = self.np.unique(x, return_counts=True)
        return vals[self.np.argmax(counts)]

    def dtype(self, x: ndarray): return x.dtype

    def where(self, condition, a: ndarray, b: ndarray): return self.np.where(condition, a, b)

    def max(self, x: ndarray, axis = 0): return self.np.max(x, axis)

    def asarray(self, x: ndarray, dtype=None, device=None): return self.np.asarray(x, dtype=dtype) 
    