from transformers import pipeline

class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        self.model = pipeline("text-classification")

    def predict(self, model_input: str):
        return self.model(model_input)
