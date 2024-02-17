import joblib
import os
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import json
from multiprocessing import Process


class BaseModel:

    DEFAULT_MODEL_PATH = "models/"

    def __init__(self, model=None, **params) -> None:
            self.model = model
            self.params = params
            self.model_name = self.generate_name()

    def __repr__(self) -> str:
        return f"{self.model_name}"
    
    def generate_name(self) -> str:
            model_name = self.model.__class__.__name__
            params_str = "_".join(f"{k}-{v}" for k, v in list(self.params.items())[:3] if v is not None)
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
        os.makedirs(path+self.model.__class__.__name__, exist_ok=True)
        full_path = os.path.join(path+self.model.__class__.__name__, self.model_name+".joblib")
        joblib.dump(self.model, filename=full_path)


    def calculate_auc(self, y_test, y_pred_proba) -> None:
        return roc_auc_score(y_test, y_pred_proba)
    
    def _plot_roc_curve(self, y_test, y_pred_proba) -> None:
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

    def plot_roc(self, y_test, y_pred_proba) -> None:
        p = Process(target=self._plot_roc_curve, args=(y_test, y_pred_proba))
        p.start()


    def make_submission(self, passenger_ids, predictions, path='results') -> None:
        submission = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': predictions.astype(int)})
        directory = os.path.join(path, self.model.__class__.__name__)
        os.makedirs(directory, exist_ok=True)
        submission_path = os.path.join(directory, 'submission.csv')
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    


    def evaluate(self, y_test, predictions, y_pred_proba, metrics=('accuracy', 'f1', 'roc_auc')):

        results = {}
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_test, predictions)
        if 'f1' in metrics:
            results['f1'] = f1_score(y_test, predictions)
        if 'roc_auc' in metrics:
            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        return results
    
    def save_stats(self, y_test, predictions, y_pred_proba=None, path='results/results.json', metrics=('accuracy', 'f1', 'roc_auc')):

        results = self.evaluate(y_test, predictions, y_pred_proba, metrics=metrics)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if os.path.exists(path):
            with open(path, 'r') as file:
                all_results = json.load(file)
        else:
            all_results = {}
        
        all_results[self.model_name] = results
        
        with open(path, 'w') as file:
            json.dump(all_results, file, indent=4)
    
    def cross_validate(self, X, y, cv=5, scoring='accuracy'):

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)  # Or any other CV strategy
        scores = cross_val_score(self.model, X, y, cv=cv_strategy, scoring=scoring)
        return scores


    def tune_hyperparameters(self, X, y, param_grid, cv=5, scoring='accuracy'):
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring=scoring, verbose=1)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        self.params.update(grid_search.best_params_)