from abc import ABC, abstractmethod
from typing import Generator

class Imputer(ABC):
    def __init__(self, model=None) -> None:
        self.model = model

    def __call__(self, data, coalitions):
        imputation_generator = self.impute(data, coalitions)
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
            outputs_pred = self.model.predict(imputed_data) # TODO: adapt to model interface
            return outputs_pred
        else:
            raise ValueError("Model must be provided for prediction")

    def postprocess(self, predictions):
        return predictions #TODO: placeholder for postprocessing
