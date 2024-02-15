import joblib
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import json

class BaseModel:

    DEFAULT_MODEL_PATH = "models/"

    def __init__(self, model=None, **params) -> None:
            self.model = model
            self.params = params
            self.model_name = self.generate_name()

    def __repr__(self) -> str:
        return f"{self.model_name}"
    def generate_name(self) -> str:
            """
            Generates a name for the model based on its class name and parameters.

            Returns:
                str: The generated name for the model.
            """
            model_name = self.model.__class__.__name__
            params_str = "_".join(f"{k}-{v}" for k, v in list(self.params.items())[:3])
            full_name = f"{model_name}_{params_str}"
            return full_name

    @classmethod
    def load_model(cls, filename=None, path=None)-> 'BaseModel':
        if path is None:
            path = cls.DEFAULT_MODEL_PATH
        if filename is None:
            model_files = [f for f in os.listdir(path) if f.startswith(cls.__name__)]
            if not model_files:
                raise FileNotFoundError(f"No model files found in '{path}' starting with {cls.__name__}.")
            filename = model_files[0]  
        full_path = os.path.join(path, filename)
        model = joblib.load(full_path)
        return cls(model=model)



    def save_model(self, path=None) -> None:
        if path is None:
            path = self.DEFAULT_MODEL_PATH
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, self.model_name+".joblib")
        joblib.dump(self.model, filename=full_path)

    def calculate_auc(self, y_test, y_pred_proba) -> None:
        return roc_auc_score(y_test, y_pred_proba)
    

    def plot_roc(self, y_test, y_pred_proba) -> None:

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f'Model (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()


    def make_submission(self, df: pd.DataFrame, predictions, path='results/submission.csv') -> None:
        submission = pd.DataFrame({'PassengerId': df['PassengerId'], 'Transported': predictions})
        os.makedirs(os.path.dirname(path)+self.model.__class__.__name__, exist_ok=True)
        submission.to_csv(path, index=False)
    
    def save_auc(self, y_test, predictions, path='results/results.json') -> None:
        auc = roc_auc_score(self.y_test, predictions)
        results = {}
        if os.path.exists(path):
            with open(path, 'r') as f:
                results = json.load(f)
                results[self.model_name] = auc
        else:
            results[self.model_name] = auc
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
    
