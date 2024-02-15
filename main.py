from src import *
import pandas as pd
shuffle = False


if __name__ == "__main__":
    PreProcess.run_all(shuffle=shuffle)
    Regression.run([LogisticRegressionModel(max_iter=1000)], ['train', 'save_model', 'plot_auc'])

    #run(shuffle=shuffle)