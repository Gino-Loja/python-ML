import pandas as pd
import numpy as np
import sklearn
import joblib

from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler  # Normalizar los datos
class Utils:
    def load_from_csv(self, path):
        return pd.read_csv(path)

    def load_from_mysql(self):
        pass

    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        #X = StandardScaler().fit_transform(X)  # Normalizamnos los datos
        #X = MeanShift().fit(X)
        return X, y

    def model_export(self, clf, score):
        print(score)
        joblib.dump(clf,
                    './models/best_model_'+str(round(score, 3))+'.pkl')
