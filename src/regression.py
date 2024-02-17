from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from .methods import LogisticRegressionModel, GradientBoostingModel
from typing import Union

train_data = pd.read_csv('data/preprocessed_train.csv')
test_data = pd.read_csv('data/preprocessed_test.csv')

Reg_Model = Union[LogisticRegressionModel, GradientBoostingModel]

class Regression:

    def __init__(self, models = {}, df : pd.DataFrame = train_data, df_test : pd.DataFrame = test_data, debug = True):

        self.df = df.copy()
        self.df.set_index('PassengerId', inplace=True, drop=True, verify_integrity=True)
        self.X = df.drop(['Transported'], axis=1, errors='ignore')
        self.y = df['Transported'].astype('int')
        self.models = models
        self.debug = debug
        self.df_test = df_test.copy()
    
    def __repr__(self):
        return f"Regression({self.models})"
    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def add_tasks(self, model : Reg_Model = None, tasks = []):
        self.models[model.generate_name()] = {'instance': model, 'tasks': tasks}

    def execute_tasks(self):

        for model_name, model_info in self.models.items():
            model : Reg_Model = model_info['instance']
            tasks = model_info['tasks']
            if 'train' in tasks:
                model.train(self.X_train, self.y_train)
                print(f"The model \"{model_name}\" has been trained.")
            if 'plot_roc' in tasks:
                model.plot_roc(self.X_test, self.y_test)
                print(f"The model \"{model_name}\" AUC has been plotted.")
            
            if 'save_model' in tasks:
                model.save_model()
                print(f"The model \"{model_name}\" has been saved.")

            if 'submission' in tasks:
                print(self.df_test.columns)
                print(self.X_test.columns)
                passenger_ids = self.df_test['PassengerId']
                #passenger_ids = self.X_test['PassengerId']
                predictions = model.predict(self.df_test)
                #predictions = model.predict(self.X_test)
                model.make_submission(passenger_ids, predictions)
            
            if 'save_stats' in tasks:
                predictions = model.predict(self.X_test)
                model.save_stats(self.y_test, predictions, metrics=['accuracy', 'f1', 'roc_auc'])

    @staticmethod
    def run(models : list = [LogisticRegressionModel()], tasks : list = ['train', 'save_model']):
        regression = Regression()
        regression.train_test_split()
        for model in models:
            regression.add_tasks(model, tasks)
        regression.execute_tasks()


