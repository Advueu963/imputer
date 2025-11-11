from jax import Array

class JaxBackend:
    import jax.numpy as jnp

    def isnan(self, x: Array): return self.jnp.isnan(x)

    def nanmean(self, x: Array, axis = 0): return self.jnp.nanmean(x, axis)

    def nanmedian(self, x: Array, axis = 0): return self.jnp.nanmean(x, axis)

    def max(self, x: Array, axis = 0): return self.jnp.max(x, axis)

    def mode(self, x: Array, axis = False): 
        vals, counts = self.jnp.unique(x, axis, return_counts=True)
        return vals[self.jnp.argmax(counts)]

    def dtype(self, x: Array): return x.dtype

    def where(self, condition, a: Array, b: Array): return self.jnp.where(condition, a, b)

    def asarray(self, x: Array, dtype=None, device=None): return self.jnp.asarray(x, dtype=dtype)
