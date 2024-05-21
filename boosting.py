import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import SGDClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Clasificador base (puede ser cualquier clasificador débil)




warnings.filterwarnings("ignore")
# if __name__ == '__main__':
#     dt_heart = pd.read_csv('./data/heart.csv')
#     #print(dt_heart['target'].describe())
#     x = dt_heart.drop(['target'], axis=1)
#     y = dt_heart['target']
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, 
#     random_state=1)
#     '''boosting = 
#     GradientBoostingClassifier(loss='exponential',learning_rate=0.15, 
#     n_estimators=188, max_depth=5).fit(X_train, y_train)
#     boosting_pred=boosting.predict(X_test)
#     print('='*64)
#     print(accuracy_score(boosting_pred, y_test))'''
#     #obtenemos el mejor resultado junto con el estimador
#     estimators = range(2, 300, 2)
#     total_accuracy = []
#     best_result = {'result' : 0, 'n_estimator': 1}
#     ada_best_result = {'result' : 0, 'n_estimator': 1}
#     base_classifier_adaboost = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train)
#     for i in estimators:
#         boost = GradientBoostingClassifier( n_estimators=i).fit(X_train, y_train)

       

# # Modelo AdaBoost
#         adaboost_model = AdaBoostClassifier(base_estimator=base_classifier_adaboost, n_estimators=i)


#         boost_pred = boost.predict(X_test)
#         adaboost_predic = adaboost_model.predict(X_test)

#         new_accuracy = accuracy_score(boost_pred, y_test)
#         adaboost_new_accuracy = accuracy_score(adaboost_predic, y_test)
#         total_accuracy.append(new_accuracy)
#         if new_accuracy > best_result['result']: 
#             best_result['result'] = new_accuracy
#             best_result['n_estimator'] = i

#         if adaboost_new_accuracy > ada_best_result['result']: 
#             ada_best_result['result'] = new_accuracy
#             ada_best_result['n_estimator'] = i
#     print(best_result)


if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/DATASET.csv')
    x = dt_heart.drop(['INCIDENCIA'], axis=1)
    y = dt_heart['INCIDENCIA']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)

    # Clasificador base (puede ser cualquier clasificador débil)
    base_classifier_adaboost = DecisionTreeClassifier(max_depth=1)

    estimators = range(2, 300, 2)
    best_result = {'result': 0, 'n_estimator': 1}
    ada_best_result = {'result': 0, 'n_estimator': 1}

    for i in estimators:
        # Crear el modelo AdaBoost para cada iteración
        adaboost_model = AdaBoostClassifier( n_estimators=i).fit(X_train, y_train)

        # Ajustar el modelo Gradient Boosting
        boost = GradientBoostingClassifier(n_estimators=i).fit(X_train, y_train)

        # Realizar predicciones para ambos modelos
        boost_pred = boost.predict(X_test)
        adaboost_predic = adaboost_model.predict(X_test)

        # Calcular la precisión para ambos modelos
        new_accuracy = accuracy_score(boost_pred, y_test)
        adaboost_new_accuracy = accuracy_score(adaboost_predic, y_test)

        # Actualizar resultados si es necesario
        if new_accuracy > best_result['result']:
            best_result['result'] = new_accuracy
            best_result['n_estimator'] = i

        if adaboost_new_accuracy > ada_best_result['result']:
            ada_best_result['result'] = adaboost_new_accuracy
            ada_best_result['n_estimator'] = i

    print("Mejor resultado para Gradient Boosting:", best_result)
    print("Mejor resultado para AdaBoost:", ada_best_result)
