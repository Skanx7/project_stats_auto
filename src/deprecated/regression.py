from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class Regression:

    def __init__(self, df : pd.DataFrame, model = None, max_iter = 10000):

        """Initializes the Regression class."""

        self.df = df.copy()
        self.X = df.drop(['Transported', 'PassengerId'], axis=1, errors='ignore')
        self.y = df['Transported'].astype('int')  # Ensure the target variable is integer encoded
        self.model = model if model is not None else LogisticRegression(max_iter=max_iter)

    def fit(self, test_size=0.2, random_state=42):

        """Fits the model to the data."""
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)

    def load_model(self, path=None):

        """Loads the model from the specified path or the first file in the 'models' directory if no path is specified."""

        if path is None:
            path = 'models/'
            files = os.listdir(path)
            if files:
                file_path = os.path.join(path, files[0])
                self.model = joblib.load(file_path)
            else:
                raise FileNotFoundError("No model files found in the 'models' directory.")
        else:
            self.model = joblib.load(path)

    def save_model(self, path):

        """Saves the model to the specified path."""

        model_name = self.model.__class__.__name__
        penalty = getattr(self.model, 'penalty', 'no_penalty')
        filename = f"{model_name}_{penalty}_model.joblib"
        full_path = os.path.join(path, filename)
        joblib.dump(self.model, filename=full_path)
        print(f"Model saved to {full_path}")

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

