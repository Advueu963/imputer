from typing import Protocol

class BackendI(Protocol):
    def isnan(self, x):
        raise NotImplementedError
    
    def nanmean(self, x, axis=0):
        raise NotImplementedError
    
    def nanmedian(self, x, axis=0):
        raise NotImplementedError
    
    def max(self, x, axis=0): 
        raise NotImplementedError
    
    def mode(self, x, axis=0, keep_dims = False): 
        raise NotImplementedError
    
    def dtype(self, x):
        raise NotImplementedError
    
    def where(self, condition, a, b):
        raise NotImplementedError
    
    def asarray(self, x, dtype=None, device=None): 
        raise NotImplementedError
    