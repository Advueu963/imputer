from abc import ABC, abstractmethod

class Imputer(ABC):
    def __init__(self, model=None) -> None:
        self.model = model

    def __call__(self, data, coalitions):
        imputed_data = self.impute(data, coalitions)
        predictions = self.predict(imputed_data)
        outputs = self.postprocess(predictions)
        return outputs

    @abstractmethod
    def impute(self, data, coalitions):
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, imputed_data):
        if self.model:
            outputs_pred = self.model.predict(imputed_data) # TODO: adapt to model interface
            return outputs_pred
        else:
            raise ValueError("Model must be provided for prediction")

    def postprocess(self, predictions):
        return predictions #TODO: placeholder for postprocessing
