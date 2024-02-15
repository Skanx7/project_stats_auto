from .plotter import Plotter
from .preprocess import PreProcess
from .regression import Regression
from .methods import LogisticRegressionModel, GradientBoostingModel
def run(shuffle = False):
    PreProcess.run_all(shuffle=shuffle)
    Regression.run()