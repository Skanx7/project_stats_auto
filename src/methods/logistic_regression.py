from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .template import BaseModel
import joblib

from typing import Optional

from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(BaseModel):
    model: Optional[LogisticRegression] = None
    """
    Logistic regression model for classification tasks.

    Parameters:
    - penalty: str, default='l2'
        The norm used in the penalization. Supported options are 'l1', 'l2', 'elasticnet', and 'none'.
    - dual: bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver.
    - tol: float, default=1e-4
        Tolerance for stopping criteria.
    - C: float, default=1.0
        Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    - fit_intercept: bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
    - intercept_scaling: float, default=1
        Useful only when the solver 'liblinear' is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a "synthetic" feature with constant value equal to intercept_scaling is appended to the instance vector.
    - class_weight: dict or 'balanced', default=None
        Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
    - random_state: int, RandomState instance, default=None
        Used when solver == 'sag', 'saga' or 'liblinear' to shuffle the data.
    - solver: {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'
        Algorithm to use in the optimization problem.
    - max_iter: int, default=100
        Maximum number of iterations taken for the solvers to converge.
    - multi_class: {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each label. For 'multinomial' the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. 'multinomial' is unavailable when solver='liblinear'. 'auto' selects 'ovr' if the data is binary, or if solver='liblinear', and otherwise selects 'multinomial'.
    - verbose: int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
    - warm_start: bool, default=False
        When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
    - n_jobs: int, default=None
        Number of CPU cores used when parallelizing over classes if multi_class='ovr'". This parameter is ignored when the solver is set to 'liblinear' regardless of whether 'multi_class' is specified or not. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
    - l1_ratio: float, default=None
        The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    
    Attributes:
    - model: The logistic regression model.

    Methods:
    - train(X_train, y_train): Trains the logistic regression model on the given training data.
    """

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False,
                 n_jobs=None, l1_ratio=None) -> None:
        
        params = {
            'penalty': penalty,
            'solver': solver,
            'dual': dual,
            'tol': tol,
            'C': C,
            'fit_intercept': fit_intercept,
            'intercept_scaling': intercept_scaling,
            'class_weight': class_weight,
            'random_state': random_state,
            'max_iter': max_iter,
            'multi_class': multi_class,
            'verbose': verbose,
            'warm_start': warm_start,
            'n_jobs': n_jobs,
            'l1_ratio': l1_ratio
        }
        super().__init__(model=LogisticRegression(**params), **params)

    def train(self, X_train, y_train):

        """
        Trains the logistic regression model on the given training data.

        Parameters:
        - X_train: The input features for training.
        - y_train: The target labels for training.
        """
        self.model.fit(X_train, y_train)
        print('Model trained successfully!')

    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Parameters:
        - X: The input data for prediction.

        Returns:
        - The predicted class labels.
        """
        return self.model.predict(X)
    def predict_proba(self, X):
        """
        Predicts the class probabilities for the given input data.

        Parameters:
        - X: The input data for prediction.

        Returns:
        - The predicted class probabilities.
        """
        return self.model.predict_proba(X)[:, 1]
    def plot_roc(self, X):
        """
        Plots the ROC curve for the model.

        Parameters:
        - y: The true target labels.
        - y_pred_proba: The predicted class probabilities.
        """
        super().plot_roc(y, self.predict_proba(X))
LogisticRegressionModel()