import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

data = pd.read_csv('seeds.csv')
features = data[data.columns[0:6]]
print(features.sample(10))

scaled_features = MinMaxScaler().fit_transform(features)
pca = PCA(n_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)
print(features_2d[0:10])

model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
km_clusters = model.fit_predict(features.values)
print(km_clusters)

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], 
                    color=colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, km_clusters)

score = silhouette_score(features.values, km_clusters)
print(f"Silhouette Score: {score:.3f}")

seed_species = data[data.columns[7]]
plot_clusters(features_2d, seed_species.values)

agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(features.values)
print(agg_clusters)

plot_clusters(features_2d, agg_clusters)

score = silhouette_score(features.values, agg_clusters)
print(f"Silhouette Score: {score:.3f}")