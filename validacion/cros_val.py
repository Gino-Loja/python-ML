import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (cross_val_score, KFold)

dataset = pd.read_csv('../data/nada.csv')
X = dataset  # .drop(['country', 'score'], axis=1)
y = dataset['Toxicos']

# print(dataset.shape)
model = DecisionTreeRegressor()
score = cross_val_score(model, X, y, cv=3,
                        scoring='neg_mean_squared_error')  # con cv podemos controlar el numOfFolds
print(score)


