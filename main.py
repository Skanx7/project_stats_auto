from src import run
from src import LogisticRegressionModel
import pandas as pd
from sklearn.model_selection import train_test_split
params = {
    'shuffle': False,
    'models': [LogisticRegressionModel(max_iter=100)],
    'tasks': ['train', 'plot_roc', 'save_model', 'submission']
}
if __name__ == "__main__":
    run(**params)