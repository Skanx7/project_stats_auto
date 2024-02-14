import xgboost as xgb
import joblib
from .template import BaseModel

class GradientBoostingModel(BaseModel):
    def __init__(self, params= None):
        super().__init__(xgb.XGBClassifier(**params) if params else xgb.XGBClassifier())

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    

