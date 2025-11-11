class Imputer():
    def __init__(self):
        pass

    def __call__(self, coalitions):
        imputed_data = self.impute(coalitions)
        predictions = self.predict(imputed_data)
        outputs = self.postprocess(predictions)
        return outputs

    @abstractmethod
    def impute(self, coalitions):
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, imputed_data):
        outputs_pred = [] # placeholder for model predictions
        return outputs_pred

    def postprocess(self, predictions):
        outputs_post = predictions # placeholder for postprocessing
        return outputs_post
