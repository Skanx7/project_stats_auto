from .preprocess import PreProcess
from .regression import Regression
from .methods import LogisticRegressionModel, GradientBoostingModel
def run(shuffle = False, models = [LogisticRegressionModel()], tasks = ['train', 'plot_roc', 'save_model']):
    PreProcess.run_all(shuffle=shuffle)
    Regression.run(models, tasks)