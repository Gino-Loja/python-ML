import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
if __name__ == "__main__":
    dataset = pd.read_csv('./data/DatasetFinal-DESKTOP-OVA2S25.csv')
    #print(dataset.head(10))
    X = dataset.fillna(0)
    #X = dataset.drop(['Project', 'Analysis', 'From', 'To','last_commit_date', '%Toxicos','Committers Weight'], axis=1)

    #Selecionamos 4 grupos
    kmeans = MiniBatchKMeans(n_clusters=3, batch_size=8).fit(X)
    print("Total de centros: " , len(kmeans.cluster_centers_))
    print("="*64)
    print(kmeans.predict(X))
    #K- means
    dataset['group'] = kmeans.predict(X)
    dataset.to_csv("DatasetFinal-gitgub.csv", index=False)
    print(dataset)
    #sns.pairplot(dataset[['bugs','lines','group']],
    #hue = 'group')
    #sns.pairplot(dataset[['temperature','rh','dew_point','wind_speed','gust_speed','wind_direction','group']], hue = 'group')
    #plt.show()
    #implementacion_k_means