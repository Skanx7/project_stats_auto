from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .template import BaseModel
import joblib

class LogisticRegressionModel(BaseModel):
    def __init__(self, max_iter=10000):
        super().__init__(LogisticRegression(max_iter=max_iter))
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
