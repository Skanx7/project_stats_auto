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
from typing import Union

Reg_Model = Union[LogisticRegressionModel, GradientBoostingModel]
class Regression:

    def __init__(self, df : pd.DataFrame, test_size = 0.2, random_state = 42):
        self.df = df.copy()
        self.X = df.drop(['Transported', 'PassengerId'], axis=1, errors='ignore')
        self.y = df['Transported'].astype('int')
        self.models = {}
    
    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def add_tasks(self, model : Reg_Model = None, tasks = []):
        self.models[model.generate_name()] = {'instance': model, 'tasks': tasks}

    def execute_tasks(self):
        for model_name, model_info in self.models.items():
            model : Reg_Model = model_info['instance']
            tasks = model_info['tasks']
            if 'train' in tasks:
                model.train(self.X, self.y)
            
            if 'plot_auc' in tasks:
                model.plot_roc(self.y, model.predict(self.X))
            
            if 'save_model' in tasks:
                model.save_model()

            if 'submission' in tasks:
                predictions = model.predict(self.df.drop('PassengerId', axis=1))
                model.make_submission(self.df, predictions)
            
            if 'save_auc' in tasks:
                predictions = model.predict(self.df.drop('PassengerId', axis=1))
                model.save_auc(self.y,)

    @staticmethod
    def run(models : list = [LogisticRegressionModel()], tasks : list = ['train', 'save_model', 'save_auc']):
        df = pd.read_csv('data/preprocessed_train.csv')
        regression = Regression(df)
        regression.train_test_split()
        for model in models:
            regression.add_tasks(model, tasks)
            print(regression.models)
        regression.execute_tasks()


