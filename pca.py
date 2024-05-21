import pandas as pd  # importamos pandas
import sklearn  # biblioteca de aprendizaje automático
import matplotlib.pyplot as plt  # Librería especializada en la creación de gráficos
from sklearn.decomposition import PCA  # importamos algorimo PCA
from sklearn.decomposition import IncrementalPCA  # importamos algorimo PCA
# clasificación y análisis predictivo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # Normalizar los datos
# permite hacer una división de un conjunto de datos en dos
from sklearn.model_selection import train_test_split
# bloques de entrenamiento y prueba de un modelo
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
# Importamos las metricas de entrenamiento y el error medio cuadrado
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # error medio cuadrado

def ipca(dt_heart):
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']  # obtenemos el target
    # KPCA
    # = dt_heart[["age", "sex", "cp", "trestbps", "chol", "fbs",
   #               "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
   # y = dt_heart[['target']]

    dt_features = StandardScaler().fit_transform(
        dt_features)  # Normalizamnos los datos

    X_train, X_test, y_train, y_test = train_test_split(
        dt_features, dt_target, test_size=0.30, random_state=42)  # 30% del conjunto de datos
    

    #KPCA
    #X_train_kpca, X_test_kpca, y_train_kpca, y_test_kpca = train_test_split(X,y, test_size=0.25)

    '''EL número de componentes es opcional, ya que por defecto si no le pasamos el número de componentes lo asignará de esta forma:
    a: n_components = min(n_muestras, n_features)'''
    # Como haremos una comparación con incremental PCA, haremos lo mismo para el IPCA.'''
    '''EL parámetro batch se usa para crear pequeños bloques, de esta forma podemos ir entrenandolos
    poco a poco y combinarlos en el resultado final'''
    ipca = IncrementalPCA(
        # tamaño de bloques, no manda a entrear todos los datos
        n_components=3, batch_size=10)
    # Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
    ipca.fit(X_train)
    ''' Aquí graficamos los números de 0 hasta la longitud de los componentes que me sugirió el PCA o que
    me generó automáticamente el pca en el eje x, contra en el eje y, el valor de la importancia
    en cada uno de estos componentes, así podremos identificar cuáles son realmente importantes
    para nuestro modelo '''
    # Ahora vamos a configurar nuestra regresión logística
    logistic = LogisticRegression(solver='lbfgs')
    # Configuramos los datos de entrenamiento
    # Mandamos los data frames la la regresión logística
    # mandasmos a regresion logistica los dos datasets

    # Calculamos nuestra exactitud de nuestra predicció

    # Configuramos los datos de entrenamiento
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train)
    # Calculamos nuestra exactitud de nuestra predicción
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))
    
if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')  # cargamos los datos
    # print(dt_heart.head(5)) #imprimimos los 5 primeros datos
    # las featurus sin el target
    #ipca(dt_heart)
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']  # obtenemos el target
    # KPCA
    # = dt_heart[["age", "sex", "cp", "trestbps", "chol", "fbs",
   #               "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
   # y = dt_heart[['target']]



    dt_features = StandardScaler().fit_transform(
        dt_features)  # Normalizamnos los datos

    X_train, X_test, y_train, y_test = train_test_split(
        dt_features, dt_target, test_size=0.30, random_state=42)  # 30% del conjunto de datos
    

    #KPCA
    #X_train_kpca, X_test_kpca, y_train_kpca, y_test_kpca = train_test_split(X,y, test_size=0.25)



    print(X_train.shape)  # consultar la forma de la tabla con pandas
    print(y_train.shape)
    '''EL número de componentes es opcional, ya que por defecto si no le pasamos el número de componentes lo asignará de esta forma:
    a: n_components = min(n_muestras, n_features)'''
    for i in range(1,10):

        pca = PCA(n_components=i)
        # Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
        pca.fit(X_train)
        # Como haremos una comparación con incremental PCA, haremos lo mismo para el IPCA.'''
        '''EL parámetro batch se usa para crear pequeños bloques, de esta forma podemos ir entrenandolos
        poco a poco y combinarlos en el resultado final'''
        ipca = IncrementalPCA(
            # tamaño de bloques, no manda a entrear todos los datos
            n_components=i, batch_size=10)
        # Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
        ipca.fit(X_train)
        ''' Aquí graficamos los números de 0 hasta la longitud de los componentes que me sugirió el PCA o que
        me generó automáticamente el pca en el eje x, contra en el eje y, el valor de la importancia
        en cada uno de estos componentes, así podremos identificar cuáles son realmente importantes
        para nuestro modelo '''

        #plt.plot(range(len(pca.explained_variance_)),
                #pca.explained_variance_ratio_)  # gneera  desde 0 hasta los componentes
        #plt.show()

        # Ahora vamos a configurar nuestra regresión logística
        logistic = LogisticRegression(solver='lbfgs')
        # Configuramos los datos de entrenamiento
        dt_train = pca.transform(X_train)  # conjunto de entrenamiento
        dt_test = pca.transform(X_test)  # conjunto de prueba
        # Mandamos los data frames la la regresión logística
        # mandasmos a regresion logistica los dos datasets
        logistic.fit(dt_train, y_train)
        # Calculamos nuestra exactitud de nuestra predicción
        print("SCORE PCA: ", logistic.score(dt_test, y_test), i)

        # Configuramos los datos de entrenamiento
        dt_train = ipca.transform(X_train)
        dt_test = ipca.transform(X_test)
        # Mandamos los data frames la la regresión logística
        logistic.fit(dt_train, y_train)
        # Calculamos nuestra exactitud de nuestra predicción
        print("SCORE IPCA: ", logistic.score(dt_test, y_test), i)

