import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

FILEPATH_TRAIN_DATA = "data/train.csv"
FILEPATH_TEST_DATA = "data/test.csv"

FILEPATH_PREPROCESSED_TRAIN_DATA = "data/preprocessed_train.csv"
FILEPATH_PREPROCESSED_TEST_DATA = "data/preprocessed_test.csv"


train_data = pd.read_csv(FILEPATH_TRAIN_DATA, sep = ",",encoding="UTF-8")
test_data = pd.read_csv(FILEPATH_TEST_DATA, sep = ",",encoding="UTF-8")


class PreProcess:

    """Class to preprocess the data for the Spaceship Titanic dataset."""
    
    ONE_HOT_ENCODED_PREFIX = "OHE_"

    
    def __init__(self, df: pd.DataFrame, is_test_data: bool = False, b_identifier_cols: bool = False, scaler = StandardScaler()):

        """Initializes the PreProcess class."""

        self.is_test_data : bool = is_test_data
        self.b_identifier_cols : bool = b_identifier_cols
        self.df : pd.DataFrame = df
        self.scaler : StandardScaler = scaler if scaler else StandardScaler()

    def set_index(self) -> None:
        pass
        #self.df.set_index("PassengerId", inplace=True, drop= True, verify_integrity=True)

    def shuffle(self) -> None:
        """Shuffles the dataframe."""
        self.df = self.df.sample(frac=1).reset_index(drop=True)
    def standardize(self) -> None:

        """Standardizes the numerical columns."""

        numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        if not self.is_test_data:
            # Fit the scaler on training data
            self.scaler.fit(self.df[numerical_columns])
        self.df.loc[:, numerical_columns] = self.scaler.transform(self.df[numerical_columns])


    def one_hot_encode(self) -> None:

        """One-hot encodes the categorical columns."""

        categorical_cols = ['CabinDeck', 'CabinSide', 'HomePlanet', 'Destination']
        prefixes = {col: f"{self.ONE_HOT_ENCODED_PREFIX}{col}_" for col in categorical_cols}
        self.df = pd.get_dummies(self.df, prefix=prefixes, prefix_sep="", columns=categorical_cols, drop_first=True)


    def feature_engineering(self) -> None:
        """Basically we just remove some features or create new ones."""
        # Splitting 'PassengerId' into 'GroupId' and 'NumberId'
        split_passenger_id = self.df['PassengerId'].str.split('_', expand=True)
        self.df.loc[:, 'GroupId'] = split_passenger_id[0]
        self.df.loc[:, 'NumberId'] = split_passenger_id[1]

        # Splitting 'Cabin' into 'CabinDeck', 'CabinNumber', and 'CabinSide'
        split_cabin = self.df['Cabin'].str.split('/', expand=True)
        self.df.loc[:, 'CabinDeck'] = split_cabin[0]
        self.df.loc[:, 'CabinNumber'] = split_cabin[1]
        self.df.loc[:, 'CabinSide'] = split_cabin[2]
        self.df.drop(columns=['Cabin'], inplace=True, errors='ignore')

        # Calculating 'FamilySize'
        family_size = self.df.groupby(['GroupId', self.df['Name'].apply(lambda x: x.split(' ')[-1] if pd.notna(x) else 'Missing')])['Name'].transform('count')
        self.df.loc[:, 'FamilySize'] = family_size

        # Handle excluded columns if specified
        if not(self.b_identifier_cols):
            excluded_cols = ['Name', 'GroupId', 'NumberId', 'CabinNumber']
            self.df.drop(columns=excluded_cols, inplace=True, errors='ignore')

    def handle_missing_values(self, method='REMOVE_NAN'):

        """Handles missing values in the dataframe with different methods."""
        if self.is_test_data:
            method = 'MODE_IMPUTATION'
        if method == 'REMOVE_NAN':
            self.df.dropna(inplace=True)
        elif method in ['MEAN_IMPUTATION', 'MEDIAN_IMPUTATION', 'MODE_IMPUTATION', 'CONSTANT_IMPUTATION']:
            strategy = {
                'MEAN_IMPUTATION': 'mean',
                'MEDIAN_IMPUTATION': 'median',
                'MODE_IMPUTATION': 'most_frequent',
                'CONSTANT_IMPUTATION': 'constant'
            }[method]

            fill_value = 0 if method == 'CONSTANT_IMPUTATION' else None
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            
            # Fit and transform imputer on the dataframe, then directly update columns
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    self.df.loc[:, col] = imputer.fit_transform(self.df[[col]])
                    
        elif method == 'KNN_IMPUTATION':
            imputer = KNNImputer(n_neighbors=5)  # Adjust n_neighbors as needed
            # Since KNNImputer returns a numpy array, we need to convert it back to a DataFrame
            imputed_array = imputer.fit_transform(self.df)
            self.df.loc[:, :] = imputed_array
        else:
            raise ValueError(f"Unknown method {method}")
            

    def run(self, method_handle_missing_values = 'REMOVE_NAN',shuffle = False) -> None:

        """Runs the preprocessing steps."""

        self.set_index()
        if shuffle:
            self.shuffle()
        self.handle_missing_values(method=method_handle_missing_values)
        self.standardize()
        self.feature_engineering()
        self.one_hot_encode()

    @staticmethod
    def run_all(shuffle = False) -> None:

        """Runs the preprocessing steps for the train and test data."""

        preprocess_train = PreProcess(train_data)
        preprocess_train.run(shuffle=shuffle, method_handle_missing_values='REMOVE_NAN')

        preprocess_test = PreProcess(test_data, is_test_data=True, scaler=preprocess_train.scaler)
        preprocess_test.run(shuffle=shuffle, method_handle_missing_values='MODE_IMPUTATION')

        preprocess_train.df.to_csv(FILEPATH_PREPROCESSED_TRAIN_DATA, encoding='utf-8', index=False)
        preprocess_test.df.to_csv(FILEPATH_PREPROCESSED_TEST_DATA, encoding='utf-8', index=False)
