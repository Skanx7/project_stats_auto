import numpy as np
import pandas as pd
import codecs

FILEPATH_TRAIN_DATA = "train.csv"
FILEPATH_TEST_DATA = "test.csv"
FILEPATH_RESULTS_DATA = "results.csv"

class DataHandler:
    """Class to handle the reading and writing of data to and from a file."""
    def __init__(self, fp : str = FILEPATH_TRAIN_DATA):
        """Initializes the DataHandler object with the file path to the data file."""
        self.fp : str = fp
        self.df : pd.DataFrame = None
        self.read()
    def read(self) -> None:
        """Reads the data from the file and stores it in the df attribute."""
        data = pd.read_csv(self.fp, sep = ",",encoding="UTF-8")
        df = pd.DataFrame(data)
        self.df = df
    def write(self, df : pd.DataFrame) -> None:
        """Writes the data to the file from the df attribute."""
        f = codecs.open(self.fp, mode='w', encoding='utf-8')
        f.write(df.to_string())
        f.close()

train_data = DataHandler(FILEPATH_TRAIN_DATA).df
test_data = DataHandler(FILEPATH_TEST_DATA).df


results = DataHandler(FILEPATH_RESULTS_DATA)

class PreProcess:
    """Class to preprocess the data for the Spaceship Titanic dataset."""
    def __init__(self, df: pd.DataFrame):
        self.redundant_cols_bool = True
        self.df : pd.DataFrame = df
    def set_index(self) -> None:
        self.df.set_index("PassengerId")
    def categorize_bools(self) -> None:
        """Categorizes the boolean columns into 0, 1, -1 for False, True, and NaN respectively."""
        bool_columns = ['CryoSleep', 'VIP', 'Transported']
        self.df[bool_columns] = self.df[bool_columns].apply(lambda x: x.map({True: 1, False: 0, np.nan: -1}))
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
        self.add_columns()

preprocess = PreProcess(train_data)
preprocess.run()
preprocessed_train_data = preprocess.df
results.write(preprocessed_train_data)

