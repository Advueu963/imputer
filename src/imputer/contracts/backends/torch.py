from torch import Tensor

class TorchBackend:
    import torch

    def isnan(self, x: Tensor): return self.torch.isnan(x)

    def nanmean(self, x: Tensor, axis = 0): return self.torch.nanmean(x, axis)

    def nanmedian(self, x: Tensor, axis = 0): return self.torch.nanmean(x, axis)

    def max(self, x: Tensor, axis = 0): return self.torch.max(x, axis)

    def mode(self, x: Tensor, axis = 0, keep_dim=False): return self.torch.mode(x, axis, keep_dim)

    def dtype(self, x: Tensor): return x.dtype

    def where(self, condition, a: Tensor, b: Tensor): return self.torch.where(condition, a, b)

    def asarray(self, x: Tensor, dtype=None, device=None):
        tensor = x if isinstance(x, self.torch.Tensor) else self.torch.as_tensor(x)
        if dtype: tensor = tensor.to(dtype=getattr(self.torch, str(dtype)) if isinstance(dtype, str) else dtype)
        if device: tensor = tensor.to(device)
        return tensor 
