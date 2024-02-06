import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs

FILEPATH_TRAIN_DATA = "data/train.csv"
FILEPATH_TEST_DATA = "data/test.csv"

FILEPATH_PREPROCESSED_TRAIN_DATA = "data/preprocessed_train.csv"
FILEPATH_PREPROCESSED_TEST_DATA = "data/preprocessed_test.csv"

FILEPATH_RESULTS_DATA = "results.csv"

class DataHandler:
    """Class to handle the reading and writing of data to and from a file."""

    def __init__(self, fp : str = FILEPATH_TRAIN_DATA):
        """Initializes the DataHandler object with the file path to the data file."""
        self.fp : str = fp
        self.read()

    def read(self) -> None:
        """Reads the data from the file and stores it in the df attribute."""
        try:
            data = pd.read_csv(self.fp, sep = ",",encoding="UTF-8")
            df = pd.DataFrame(data)
            self.df = df
        except:
            self.df = None

    def write(self, df : pd.DataFrame) -> None:
        """Writes the data to the file from the df attribute."""
        df.to_csv(self.fp, encoding='utf-8', index=False)


train_data = DataHandler(FILEPATH_TRAIN_DATA).df
test_data = DataHandler(FILEPATH_TEST_DATA).df

train_prep = DataHandler(FILEPATH_PREPROCESSED_TRAIN_DATA)
test_prep = DataHandler(FILEPATH_PREPROCESSED_TEST_DATA)
results = DataHandler(FILEPATH_RESULTS_DATA)


class PreProcess:
    """Class to preprocess the data for the Spaceship Titanic dataset."""

    def __init__(self, df: pd.DataFrame, is_test_data: bool = False, redundant_cols_bool: bool = False):
        self.is_test_data = is_test_data
        self.redundant_cols_bool = redundant_cols_bool
        self.df : pd.DataFrame = df

    def set_index(self) -> None:
        self.df.set_index("PassengerId")

    def categorize_bools(self) -> None:
        """Categorizes the boolean columns into 0, 1 for False, True"""
        bool_columns = ['CryoSleep', 'VIP'] + (1-self.is_test_data)*['Transported']
        self.df[bool_columns] = self.df[bool_columns].apply(lambda x: x.map({True: int(1), False: int(0)}))

    def standardize(self) -> None:
        """Standardizes the numerical columns."""
        numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']  # Adjust as needed

        for column in numerical_columns:
            mean = self.df[column].mean()
            std = self.df[column].std()
            self.df[column] = (self.df[column] - mean) / std


    def add_columns(self) -> None:
        """Adds new columns to the dataframe."""
        self.df[['GroupId', 'NumberId']] = self.df['PassengerId'].str.split('_', expand=True)                   # Split the PassengerId into two columns
        self.df[['CabinDeck', 'CabinNumber', 'CabinSide']] = self.df['Cabin'].str.split('/', expand=True)       # Split the Cabin into three columns
        excluded_cols = [] if self.redundant_cols_bool else ['PassengerId', 'Cabin']
        cols = ['GroupId', 'NumberId', 'CabinDeck', 'CabinNumber', 'CabinSide'] +\
            [col for col in self.df.columns if col not in ['GroupId', 'NumberId', 'CabinDeck', 'CabinNumber', 'CabinSide'] and col not in excluded_cols]
        self.df = self.df[cols]

    def run(self) -> None:
        """Runs the preprocessing steps."""
        self.set_index()
        self.categorize_bools()
        self.standardize()
        self.add_columns()
        
    @staticmethod
    def run_all() -> None:
        """Runs the preprocessing steps for the train and test data."""
        preprocess_train = PreProcess(train_data)
        preprocess_train.run()
        preprocess_test = PreProcess(test_data, True)
        preprocess_test.run()
        train_prep.write(preprocess_train.df)
        test_prep.write(preprocess_test.df)

class Plotter:
    """Class to plot the data for the Spaceship Titanic dataset."""
    def __init__(self, df: pd.DataFrame):
        self.df : pd.DataFrame = df

    def plot(self) -> None:
        """Plots the data using subplots."""
        columns_to_plot = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        nrows = len(columns_to_plot) // 2 + len(columns_to_plot) % 2 
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(15, nrows * 5))

        axs = axs.flatten()
        
        for i, column in enumerate(columns_to_plot):
            transported = self.df[self.df['Transported'] == True][column]
            not_transported = self.df[self.df['Transported'] == False][column]
            
            axs[i].hist(transported, alpha=0.5, label='Transported', bins=20, edgecolor='black')
            axs[i].hist(not_transported, alpha=0.5, label='Not Transported', bins=20, edgecolor='black')
            
            axs[i].set_title(f'Histogram of {column} by Transported Status')
            axs[i].set_xlabel(column)
            axs[i].set_ylabel('Frequency')
            axs[i].legend()

        plt.tight_layout()

        plt.show()
        
        if len(columns_to_plot) % 2 != 0:
            axs[-1].axis('off')


PreProcess.run_all()

plotter = Plotter(train_prep.df)
plotter.plot()

