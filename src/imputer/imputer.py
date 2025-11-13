from abc import ABC, abstractmethod
from typing import Generator, Optional, Union
import numpy as np
import re
import lazy_dispatch as ld

from imputer.torch_data_loader import get_torch_dataloader
from imputer.base_data_loader import base_data_loader

class Imputer(ABC):
    def __init__(self, model=None, backend="") -> None:
        self.model = model
        self.backend = backend

    @ld.lazydispatch
    def expand_coalitions(data: object, coalitions: object, feature_group: object):
        return None
    
    @expand_coalitions.register(np.ndarray)
    def expand_coalitions_np(data: np.ndarray, coalitions: np.ndarray, feature_group: Union[dict[int, str], str]) -> np.ndarray:
        if isinstance(feature_group, str):
            # Parse grid dimensions from string (e.g., "3x3" or "3x3x3")
            grid_dims = tuple(map(int, feature_group.lower().split('x')))
            num_dims = len(grid_dims)

            if len(data.shape) == 2:
                num_features = data.shape[1]
                feature_dim_per_axis = int(num_features ** (1.0 / num_dims))
                feature_dims = tuple([feature_dim_per_axis] * num_dims)
            else:
                feature_dims = data.shape[1:1+num_dims]
                        
            expand_factors = tuple(feature_dims[i] // grid_dims[i] for i in range(num_dims))
            
            expanded_coalitions = []
            for coalition in coalitions:
                # Reshape coalition to grid and expand each dimension
                expanded = coalition.reshape(grid_dims)
                for axis, factor in enumerate(expand_factors):
                    expanded = np.repeat(expanded, factor, axis=axis)
                expanded_coalitions.append(expanded.flatten())
            
            return np.array(expanded_coalitions)
        else:
            expanded_coalitions = np.full(data.shape, None, dtype=object)
        
            for k, v in feature_group.items():
                index = k
                if isinstance(index, int):
                    index = [index]
                coalition_values = [coalitions[idx] for idx in index]
                parts = re.split(r'[,;:]', v)
                parts = [part.strip() for part in parts if part.strip()]
                if parts[-1] != ',':
                    parts.append([","])
                current_index = []
                for i, part in enumerate(parts):
                    if part in [',', ';', ':']:
                        continue
                    parts[i] = int(part)
                for i, part in enumerate(parts):
                    if part != ':':
                        continue
                    prev_part = parts[i - 1] if i > 0 else None
                    next_part = parts[i + 1] if i + 1 < len(parts) else None
                    if prev_part is None or next_part is None or not (isinstance(prev_part, int) and isinstance(next_part, int)):
                        raise ValueError("Invalid feature group format around ':'.")
                    combined = [x for x in range(prev_part, next_part + 1)]
                    parts[i] = combined

                for i, part in enumerate(parts):
                    if part != ';':
                        continue
                    prev_part = parts.pop(i - 1) if i > 0 else None
                    next_part = parts.pop(i + 1) if i + 1 < len(parts) else None
                    if prev_part is None or next_part is None:
                        raise ValueError("Invalid feature group format around ';'.")
                    combined = []
                    for prev in prev_part:
                        if isinstance(prev, int):
                            prev = [prev]
                        for nex in next_part:
                            if isinstance(nex, int):
                                nex = [nex]
                            combined.append(prev.append(nex)) # type: ignore
                    parts[i] = combined
                for i, part in enumerate(parts):
                    if part == ',':
                        continue
                    for idx in part:
                        if isinstance(idx, int):
                            expanded_coalitions[idx] = coalition_values
                        elif isinstance(idx, list):
                            expanded_coalitions[tuple(idx)] = coalition_values

        if np.any(expanded_coalitions == None):
            none_positions = np.argwhere(expanded_coalitions == None)
            raise ValueError(f"Expanded coalitions contain None values at positions: {none_positions.tolist()}, the feature_group mapping was incomplete.")
                    
        return coalitions

    def __call__(self, data, coalitions, feature_group: Optional[Union[dict[int, str], str]] = None):
        if feature_group is not None:
            coalitions_processed = self.expand_coalitions(  data, coalitions, feature_group) # type: ignore
        else:
            coalitions_processed = coalitions
        imputation = (lambda a: self.impute(a, coalitions_processed))
        if self.backend == "torch":
            imputation_generator = get_torch_dataloader(data, imputation)
        else:
            imputation_generator = base_data_loader(data, imputation)
        outputs = []
        for i, imputed_data in enumerate(imputation_generator):
            print(f"[Imputer __call__]: Imputed data for coalition {i}: {imputed_data}")
            predictions = self.predict(imputed_data)
            outputs.append(self.postprocess(predictions))
        return outputs

    @abstractmethod
    def impute(self, data, coalitions):
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, imputed_data):
        if self.model:
            outputs_pred = self.predict_model(self.model, imputed_data) # TODO: adapt to model interface
            return outputs_pred
        else:
            raise ValueError("Model must be provided for prediction")

    def postprocess(self, predictions):
        return predictions #TODO: placeholder for postprocessing
    
    @ld.lazydispatch
    def predict_model(model: object, data: object):
        return data

    @predict_model.register("sklearn.base.BaseEstimator")
    def predict_sklearn_model(model: "sklearn.base.BaseEstimator", data: object):
        return model.predict(data)
    
    @predict_model.register("torch.nn.Module")
    def predict_torch_model(model: "torch.nn.Module", data: object):
        import torch
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        return model(data)
    
    @predict_model.register("flax.nnx.Module")
    def predict_flax_model(model: "flax.nnx.Module", data: object):
        return model(data)
