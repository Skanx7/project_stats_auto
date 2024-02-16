from src import *
import pandas as pd
shuffle = False
from sklearn.model_selection import train_test_split



PreProcess.run_all(shuffle=shuffle)
df = pd.read_csv('data/preprocessed_train.csv')
X = df.drop(['Transported', 'PassengerId'], axis=1, errors='ignore')
y = df['Transported'].astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lm = LogisticRegressionModel(max_iter=10000)
print(lm)
lm.train(X_train,y_train)
lm.plot_roc(X_test, y_test)