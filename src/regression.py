from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Regression:
    def __init__(self, df):
        self.df = df
        self.X = df.drop(['Transported', 'Name', 'PassengerId', 'GroupId', 'NumberId', 'CabinNumber'], axis=1, errors='ignore')  # Drop non-numeric and target columns
        self.y = df['Transported']

    def preprocess(self):
        # Handling categorical variables (OneHotEncoding or similar)
        categorical_cols = ['CabinDeck', 'CabinSide', 'HomePlanet', 'Destination']
        self.X = pd.get_dummies(self.X, columns=categorical_cols)
        # Scaling numerical columns if not already done (optional, as your data seems already standardized)
        
    def fit(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # Initialize and fit the Logistic Regression model
        self.model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test

    def plot_roc(self):
        # Predict probabilities
        probs = self.model.predict_proba(self.X_test)[:, 1]
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc_score(self.y_test, probs))
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

# Example usage:
# Load your dataframe
df = pd.read_csv('preprocessed_train.csv')
regression = Regression(df)
regression.preprocess()
regression.fit()
regression.plot_roc()
