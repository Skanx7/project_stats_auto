import joblib
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

class BaseModel:

    DEFAULT_MODEL_PATH = "models/"

    def __init__(self, model=None, **params):
            self.model = model
            self.params = params
            self.model_filename = self.generate_filename()

    def generate_filename(self):
        model_name = self.model.__class__.__name__
        params_str = "_".join(f"{k}-{v}" for k, v in list(self.params.items())[:3])
        filename = f"{model_name}_{params_str}.joblib"
        return filename

    @classmethod
    def load_model(cls, filename=None, path=None):
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



    def save_model(self, path=None):
        if path is None:
            path = self.DEFAULT_MODEL_PATH
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, self.model_filename)
        joblib.dump(self.model, filename=full_path)


    def calculate_auc(self, y_test, y_pred_proba):
        return roc_auc_score(y_test, y_pred_proba)
    

    def plot_roc(self, y_test, y_pred_proba):

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


    def make_submission(self, df: pd.DataFrame, predictions, path='results/submission.csv'):
        submission = pd.DataFrame({'PassengerId': df['PassengerId'], 'Transported': predictions})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        submission.to_csv(path, index=False)
        print(f"Submission file written to {path}")
