from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from .methods import LogisticRegressionModel, GradientBoostingModel

class Regression:

    def __init__(self, df : pd.DataFrame):
        self.df = df.copy()
        self.X = df.drop(['Transported', 'PassengerId'], axis=1, errors='ignore')
        self.y = df['Transported'].astype('int')
        self.models = {}
    
    def add_task(self, model : LogisticRegressionModel | GradientBoostingModel = None, tasks = []):
        self.models[model.__class__.__name__] = {'instance': model, 'tasks': tasks}

    def execute_tasks(self):
        for model_name, model_info in self.models.items():
            model = model_info['instance']
            tasks = model_info['tasks']

            if 'train' in tasks:
                self._train_model(model, model_name)
            
            if 'plot_auc' in tasks:
                self._plot_roc(model, model_name)
            
            if 'submission' in tasks:
                self._make_submission(model, model_name)
            
            if 'add_auc_csv' in tasks:
                self._add_auc_to_csv(model, model_name)


    def fit(self, test_size=0.2, random_state=42):

        """Fits the model to the data."""
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)

    def plot_roc(self):
        
        """Plots the ROC curve for the model."""

        probs = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Logistic Regression (area = {roc_auc_score(self.y_test, probs):0.2f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    def predict(self, df : pd.DataFrame) -> np.ndarray:

        """Predicts the target variable using the model."""

        return self.model.predict(df)
    
    def make_submission(self, df : pd.DataFrame, path='results/') -> None:

        """Makes a submission file from the predictions of the model."""

        submission = pd.DataFrame({'PassengerId': df['PassengerId'], 'Transported': self.predict(df.drop('PassengerId', axis=1))})
        submission.to_csv(os.path.join(path, 'submission.csv'), index=False)

    @staticmethod
    def run():
        df = pd.read_csv('data/preprocessed_train.csv')
        regression_model = Regression(df)
        regression_model.fit()
        regression_model.save_model(path='models/')
        regression_model.plot_roc()
        print("Model saved to 'models/' directory.")

