from cog import BasePredictor, Input, Path
from transformers import pipeline

class Predictor(BasePredictor):
    def setup(self):
        self.model = None
        self.model = pipeline("text-classification")

    def predict(self,
          model_input: str
    ) -> str:
        return self.model(model_input)
