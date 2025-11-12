from abc import ABC, abstractmethod
from typing import Generator, Optional, Union
import numpy as np
import lazy_dispatch as ld

class Imputer(ABC):
    def __init__(self, model=None, feature_group: Optional[Union[dict[int, str], str]] = None) -> None:
        self.model = model

    @ld.lazydispatch
    def expand_coalitions(coalitions: object, feature_group: object):
        return None
    
    @expand_coalitions.register(np.ndarray)
    def expand_coalitions_np(coalitions: np.ndarray, feature_group: Union[dict[int, str], str]) -> np.ndarray:
        if isinstance(feature_group, str):
            raise NotImplementedError("String-based feature groups are not implemented yet.")
        else:
            raise NotImplementedError("Dict-based feature groups are not implemented yet.")

    def __call__(self, data, coalitions, feature_group: Optional[Union[dict[int, str], str]] = None):
        if feature_group:
            coalitions_processed = self.expand_coalitions(coalitions, feature_group=feature_group)
        else:
            coalitions_processed = coalitions
        imputation_generator = self.impute(data, coalitions_processed)
        outputs = []
        for i, imputed_data in enumerate(imputation_generator):
            print(f"[Imputer __call__]: Imputed data for coalition {i}: {imputed_data}")
            predictions = self.predict(imputed_data)
            outputs.append(self.postprocess(predictions))
        return outputs

    @abstractmethod
    def impute(self, data, coalitions) -> Generator:
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, imputed_data):
        if self.model:
            outputs_pred = self.predict_model(self.model, imputed_data)# TODO: adapt to model interface
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
        return model(data)
    
    @predict_model.register("flax.nnx.Module")
    def predict_flax_model(model: "flax.nnx.Module", data: object):
        return model(data)
