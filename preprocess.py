import pandas as pd
import codecs

FILEPATH_TRAIN_DATA = "train.csv"
FILEPATH_TEST_DATA = "test.csv"
FILEPATH_RESULTS_DATA = "results.csv"

class DataHandler:
    def __init__(self, fp : str = FILEPATH_TRAIN_DATA):
        self.df : pd.DataFrame = None
        self.fp : str = fp
    def read(self, fp : str = FILEPATH_TRAIN_DATA) -> pd.DataFrame:
        data = pd.read_csv(fp, sep = ",",encoding="UTF-8")
        df = pd.DataFrame(data)
        self.df = df
    def write(self, df : pd.DataFrame) -> None:
        f = codecs.open(self.fp, mode='w', encoding='utf-8')
        f.write(df.to_string())
        f.close()
class PreProcess:
    def __init__(self, dh: DataHandler):
        self.dh = dh
        

def write_data(df: pd.DataFrame, fp : str = FILEPATH_RESULTS_DATA) -> None:
    f = codecs.open(fp, mode='w', encoding='utf-8')
    f.write(df.to_string())
    f.close()
def set_index(df: pd.DataFrame, fp: str = FILEPATH_TRAIN_DATA) -> None:
    df.set_index("PassengerId")
def add_columns(df: pd.DataFrame) -> None:
    #We seperate the group and number in the group and the number in the group into gggg and pp columns instead of gggg_pp which are unique
    passenger_ids : list = df.loc[:, 'PassengerId']
    splitted_passenger_ids = [passenger_ids[i].split('_') for i in range(len(passenger_ids))]
    group_ids = map(lambda c : c[0], splitted_passenger_ids)
    number_ids = map(lambda c : c[0], splitted_passenger_ids)
    df.insert(0,group_ids)

df = read_data()
add_columns(df)
write_data(df)
