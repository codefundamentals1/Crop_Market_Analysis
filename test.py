from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeRegressor

def train(csv_name):
    
    dataset = pd.read_csv(csv_name)
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 3].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=69)

    # Fitting decision tree regression to dataset
    depth = random.randrange(7,18)
    regressor = DecisionTreeRegressor(max_depth=depth)
    regressor.fit(X_train, Y_train)
    y_pred_tree = regressor.predict(X_test)
    print(y_pred_tree)
    print(Y_test)
    #fsa=np.array([float(1),2019,45]).reshape(1,3)
    #fask=regressor_tree.predict(fsa)

train("static/Wheat.csv")