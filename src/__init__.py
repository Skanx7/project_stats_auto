from .plotter import Plotter
from .preprocess import PreProcess
from .regression import Regression
def run(shuffle = False):
    PreProcess.run_all(shuffle=shuffle)
    Regression.run()