import lazy_dispatch as ld
import numpy as np
from imputer.imputer import Imputer

class AttentionImputer(Imputer):
    def __init__(self, model=None, backend=""):
        super().__init__(model, backend)

    def impute(self, data, coalitions) -> None:
        return self.impute_attention(data, coalitions)

    @ld.lazydispatch
    def impute_attention(data: object, coalitions:object) -> object:
        return None
    
    @impute_attention.register("np.ndarray")
    def impute_attention_numpy(data: "np.ndarray", coalitions:"np.ndarray") -> "np.ndarray":
        ret = np.zeros((coalitions.shape[0], data.shape[0]))
        for i in range(coalitions.shape[0]):
            ret[i] = data * coalitions[i]
        return ret
    
    @impute_attention.register("torch.Tensor")
    def impute_attention_torch(data: "torch.Tensor", coalitions: "torch.Tensor") -> "torch.Tensor":
        import torch
        ret = torch.zeros(coalitions.shape[0], data.shape[0])
        for i in range(coalitions.shape[0]):
            ret[i] = data[i] * coalitions[i]
        return ret
