from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .template import BaseModel
import joblib
import numpy as np

from typing import Optional

from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(BaseModel):
    model: Optional[LogisticRegression]
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False,
                 n_jobs=None, l1_ratio=None) -> None:
        
        params = {
            'penalty': penalty,
            'solver': solver,
            'max_iter': max_iter,
            'dual': dual,
            'tol': tol,
            'C': C,
            'fit_intercept': fit_intercept,
            'intercept_scaling': intercept_scaling,
            'class_weight': class_weight,
            'random_state': random_state,
            'multi_class': multi_class,
            'verbose': verbose,
            'warm_start': warm_start,
            'n_jobs': n_jobs,
            'l1_ratio': l1_ratio
        }
        super().__init__(model=LogisticRegression(**params), **params)

    def train(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
    def plot_roc(self, X, y) -> None:
        super().plot_roc(y, self.predict_proba(X)[:,1])