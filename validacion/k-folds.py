import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (
    cross_val_score, KFold
)
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


import warnings
warnings.simplefilter("ignore")

dataset = pd.read_csv('../data/nada.csv')
X = dataset.drop(['Toxicos'], axis=1)
y = dataset['Toxicos']
# print(dataset.shape)
# model = DecisionTreeRegressor()
# score = cross_val_score(model, X, y, cv=3,
#                         scoring='neg_mean_squared_error')  # con cv podemos controlar el

# print(score)

#print(np.abs(np.mean(score)))
# kf = KFold(n_splits=3, shuffle=True, random_state=42)
# mse_values = []
# for train, test in kf.split(dataset):

#     X_train = X.iloc[train]
#     y_train = y.iloc[train]
#     X_test = X.iloc[test]
#     y_test = y.iloc[test]
#     model = DecisionTreeRegressor().fit(X_train, y_train)
#     predict = model.predict(X_test)
#     mse_values.append(mean_squared_error(y_test, predict))
#     print("Los tres MSE fueron: ", mse_values)
    
#     print("El MSE promedio fue: ", np.mean(mse_values))


class Models:
    def __init__(self):
        self.reg = {
             'LinearSVC': LinearSVC(),
             'GradientClass' : GradientBoostingClassifier(),
             'RandomForest': RandomForestClassifier(),
             'KNeighbors': KNeighborsClassifier()
        }
        self.params = {
            'LinearSVC': {
                'max_iter' : [1000],
            },
            'GradientClass' : {
                'n_estimators' : [150],
                'learning_rate': [0.01, 0.05, 0.1],
                'criterion': ['friedman_mse', 'mse']
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            },
            'KNeighbors': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    def grid_training(self, X, y):
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            grid_reg = GridSearchCV(reg, self.params[name], cv = 3).fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)
            if score > best_score:
                best_score = score
                best_model = grid_reg.best_estimator_
        print(best_score)




class ModelsRam:
    def __init__(self):
        self.reg = {
            'LinearSVC': LinearSVC(),
            'GradientClass': GradientBoostingClassifier(),
            'RandomForest': RandomForestClassifier(),
            'KNeighbors': KNeighborsClassifier()
        }
        self.params = {
            'LinearSVC': {
                'max_iter': [1000],
            },
            'GradientClass': {
                'n_estimators': [150],
                'learning_rate': [0.01, 0.05, 0.1],
                'criterion': ['friedman_mse', 'mse']
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            },
            'KNeighbors': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    
    def grid_training(self, X, y):
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            random_reg = RandomizedSearchCV(reg, self.params[name], cv=3).fit(X, y.values.ravel())
            score = np.abs(random_reg.best_score_)
            if score > best_score:
                best_score = score
                best_model = random_reg.best_estimator_
        print(score)
model = ModelsRam()
model.grid_training(X, y)