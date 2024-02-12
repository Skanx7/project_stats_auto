import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data_handler import train_data, test_data, train_prep, test_prep, results
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        scaler = StandardScaler()
        self.df[numerical_columns] = scaler.fit_transform(self.df[numerical_columns])

    def add_columns(self) -> None:
        """Adds new columns to the dataframe."""
        self.df[['GroupId', 'NumberId']] = self.df['PassengerId'].str.split('_', expand=True)                   # Split the PassengerId into two columns
        self.df[['CabinDeck', 'CabinNumber', 'CabinSide']] = self.df['Cabin'].str.split('/', expand=True)       # Split the Cabin into three columns
        self.df['FamilySize'] = self.df.groupby(['GroupId', self.df['Name'].apply(lambda x: x.split(' ')[-1] if pd.notna(x) else 'Missing')])['Name'].transform('count')
        # One-hot encode the categorical columns 
        # basically we just add a binary column for each unique category
        categorical_cols = ['CabinDeck', 'CabinSide', 'HomePlanet', 'Destination']
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
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



