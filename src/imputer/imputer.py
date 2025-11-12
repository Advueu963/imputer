from abc import abstractmethod

class Imputer():
    def __init__(self) -> None:
        pass

    def __call__(self, coalitions, model=None):
        imputed_data = self.impute(coalitions)
        predictions = self.predict(imputed_data, model=model)
        outputs = self.postprocess(predictions)
        return outputs

    @abstractmethod
    def impute(self, coalitions):
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, imputed_data, model):
        if model:
            outputs_pred = model.predict(imputed_data) # TODO: adapt to model interface
            return outputs_pred
        else:
            raise ValueError("Model must be provided for prediction")

    def postprocess(self, predictions):
        return predictions #TODO: placeholder for postprocessing
