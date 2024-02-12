import pandas as pd

FILEPATH_TRAIN_DATA = "data/train.csv"
FILEPATH_TEST_DATA = "data/test.csv"

FILEPATH_PREPROCESSED_TRAIN_DATA = "data/preprocessed_train.csv"
FILEPATH_PREPROCESSED_TEST_DATA = "data/preprocessed_test.csv"

FILEPATH_RESULTS_DATA = "results/results.csv"

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