import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer

import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/DATASET.csv')
    # print(dt_heart['target'].describe())
    x = dt_heart.drop(['INCIDENCIA'], axis=1)
    y = dt_heart['INCIDENCIA']

    x = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform').fit_transform(x)


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35,
                                                        random_state=1)
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_prediction = knn_class.predict(X_test)
    print('='*64)
    print('SCORE con KNN: ', accuracy_score(knn_prediction, y_test))
    '''bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(),
    n_estimators=50).fit(X_train, y_train) # base_estimator pide el estimador en el 
    que va a estar basado nuestro metodo || n_estimators nos pide cuantos de 
    estos modelos vamos a utilizar
    bag_pred = bag_class.predict(X_test)
    print('='*64)
    print(accuracy_score(bag_pred, y_test))'''

    # SE TENGA QUE USAR EN APRENDIZAJE SUPERVISADO
    estimators = {
        'LogisticRegression': LogisticRegression(),  # 10
        'SVC': SVC(),  # 10
        'LinearSVC': LinearSVC(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
        'KNN': KNeighborsClassifier(),
        'DecisionTreeClf': DecisionTreeClassifier(),
        'RandomTreeForest': RandomForestClassifier(random_state=0)

    }
    estimators_mayor = {
        'LogisticRegression': {"n_estimators": 0,
                               "valor": 0},
        'SVC': {"n_estimators": 0,
                "valor": 0},
        'LinearSVC': {"n_estimators": 0,
                      "valor": 0},
        'SGD': {"n_estimators": 0,
                "valor": 0},
        'KNN': {"n_estimators": 0,
                "valor": 0},
        'DecisionTreeClf': {"n_estimators": 0,
                            "valor": 0},
        'RandomTreeForest': {"n_estimators": 0,
                             "valor": 0} # Nuevo algoritmo

    }
    # lista = []
    estimators_item = range(2, 100, 2)
    for name, estimator in estimators.items():
        for i in estimators_item:
            bag_class = BaggingClassifier(base_estimator=estimator,
                                          n_estimators=i).fit(X_train, y_train)
            bag_predict = bag_class.predict(X_test)
            # print('='*64)
            # print('SCORE Bagging with {} : {}'.format(name,
            # accuracy_score(bag_predict, y_test)))
            if accuracy_score(bag_predict, y_test) > estimators_mayor[name]["valor"]:
                estimators_mayor[name]["valor"] = accuracy_score(
                    bag_predict, y_test)
                estimators_mayor[name]["n_estimators"] = i

    # print(estimators_mayor)
    for name, estimator in estimators_mayor.items():
        print(name, estimator)
