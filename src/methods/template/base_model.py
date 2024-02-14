import joblib
import os

class BaseModel:
    def __init__(self, model=None):

        self.model = model

    def save_model(self, path):

        """Saves the model to the specified path."""

        model_name = self.model.__class__.__name__
        penalty = getattr(self.model, 'penalty', 'no_penalty')
        filename = f"{model_name}_{penalty}.joblib"
        full_path = os.path.join(path, filename)
        joblib.dump(self.model, filename=full_path)

    def load_model(self, path):

        """Loads the model from the specified path."""

        self.model = joblib.load(path)