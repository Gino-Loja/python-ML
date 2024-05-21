import pandas as pd #importamos pandas
import sklearn #biblioteca de aprendizaje automático
import matplotlib.pyplot as plt #Librería especializada en la creación de gráficos
from sklearn.decomposition import PCA #importamos algorimo PCA
from sklearn.decomposition import IncrementalPCA #importamos algorimo PCA
from sklearn.linear_model import LogisticRegression #clasificación y análisis predictivo 
from sklearn.preprocessing import StandardScaler #Normalizar los datos
from sklearn.model_selection import train_test_split #permite hacer una división de un conjunto de datos en dos 
#bloques de entrenamiento y prueba de un modelo

if __name__ == '__main__':

    #carga de datos
# Leer los datos desde los archivos CSV
    df_manual = pd.read_csv('./data/Reporte_Monitoreo_2022_12_30.csv')

# Cargar datos del sensor 
    df_sensor = pd.read_csv('./data/Datos_iniap_ordenados.csv')


    """
    df_manual['Fecha'] = pd.to_datetime('2022-12-30') 
    # Convertir fecha del sensor a datetime
    df_sensor['Date'] = pd.to_datetime(df_sensor['Date'], format='%m/%d/%y %H:%M:%S')


    # Resumir datos del sensor por día
    
    df_sensor_daily = df_sensor.groupby(df_sensor['Date']).mean().reset_index()

    # Renombrar columnas
    df_sensor_daily.columns = ['Fecha', 'Humedad', 'Temperatura', 'Punto Rocio', 'Viento', 'Rafagas', 'Direccion Viento', 'Lluvia']

    # Unir dataframes por fecha y planta
    df_merged = pd.merge(df_manual, df_sensor_daily, how='left', left_on='Fecha', right_on='Fecha')
    # Rellenar NA
    df_merged = df_merged.fillna(0)

    # Quedarse sólo con columnas de interés
    df_merged = df_merged[['PLANTA', 'FRUTO','INCIDENCIA', 'SEVERIDAD (%)', 'Humedad', 'Temperatura', 'Punto Rocio', 'Viento', 'Rafagas', 'Direccion Viento', 'Lluvia']]
    """

    df_sensor['Date'] = pd.to_datetime(df_sensor['Date'], format='%m/%d/%Y %H:%M',errors='coerce')
    df_merged = pd.merge(df_manual, df_sensor, left_on=['PLANTA'], right_on=[df_sensor['Date'].dt.strftime('%m/%d/%Y')])

# Eliminar columnas duplicadas y la columna redundante 'Date_y'
    df_merged = df_merged.drop(['Date_y', 'PLANTA'], axis=1)

# Renombrar columnas para mayor claridad
    df_merged = df_merged.rename(columns={'Date_x': 'Date'})

    df_merged.to_csv('datos_fusionados.csv', index=False)


"""



    dt_heart=pd.read_csv('./data/heart.csv') #cargamos los datos
    #print(dt_heart.head(5)) #imprimimos los 5 primeros datos
    dt_features=dt_heart.drop(['target'],axis=1) #las featurus sin el target
    dt_target = dt_heart['target'] #obtenemos el target
    
    dt_features = StandardScaler().fit_transform(dt_features) #Normalizamnos los datos
    
    X_train,X_test,y_train,y_test =train_test_split(dt_features,dt_target,test_size=0.30,random_state=42) #30% del conjunto de datos
    print(X_train.shape) #consultar la forma de la tabla con pandas
    print(y_train.shape)
    '''EL número de componentes es opcional, ya que por defecto si no le pasamos el número de componentes lo asignará de esta forma:
    a: n_components = min(n_muestras, n_features)'''
    pca=PCA(n_components=3)
    # Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
    pca.fit(X_train)
     #Como haremos una comparación con incremental PCA, haremos lo mismo para el IPCA.'''
    '''EL parámetro batch se usa para crear pequeños bloques, de esta forma podemos ir entrenandolos
    poco a poco y combinarlos en el resultado final'''
    ipca=IncrementalPCA(n_components=3,batch_size=10) #tamaño de bloques, no manda a entrear todos los datos
    #Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
    ipca.fit(X_train)
    ''' Aquí graficamos los números de 0 hasta la longitud de los componentes que me sugirió el PCA o que
    me generó automáticamente el pca en el eje x, contra en el eje y, el valor de la importancia
    en cada uno de estos componentes, así podremos identificar cuáles son realmente importantes
    para nuestro modelo '''
    plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_ratio_) #gneera  desde 0 hasta los componentes
    plt.show()
    #Ahora vamos a configurar nuestra regresión logística
    logistic=LogisticRegression(solver='lbfgs')
    # Configuramos los datos de entrenamiento
    dt_train = pca.transform(X_train)#conjunto de entrenamiento
    dt_test = pca.transform(X_test)#conjunto de prueba
     # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train) #mandasmos a regresion logistica los dos datasets
    #Calculamos nuestra exactitud de nuestra predicción
    print("SCORE PCA: ", logistic.score(dt_test, y_test))
    
    #Configuramos los datos de entrenamiento
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train)
    #Calculamos nuestra exactitud de nuestra predicción
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))

    """