"""
Created on Tue Nov  7 22:41:16 2023

@author: Bren Guzmán
"""
#%% LIBRERÍAS

from Dataframe import Dataframe
from KMeans import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#%% CARGAR LOS DATOS

#%%% X Train

# Ruta de los archivos
training_images_filepath = 'MNIST/train-images.idx3-ubyte'
training_labels_filepath = 'MNIST/train-labels.idx1-ubyte'

# Crear una instancia de la clase Dataframe
train_dataframe = Dataframe(training_images_filepath, training_labels_filepath)

# Cargar los datos
x_train, y_train = train_dataframe.load_data()



#%%% X Test

# Ruta de los archivos
test_images_filepath = 'MNIST/t10k-images.idx3-ubyte'
test_labels_filepath = 'MNIST/t10k-labels.idx1-ubyte'

# Crear una instancia de la clase Dataframe
test_dataframe = Dataframe(test_images_filepath, test_labels_filepath)

# Cargar los datos
x_test, y_test = test_dataframe.load_data()


#%% IMPLEMENTACIÓN K-MEANS

#%%% Fit con x_train

kmeans = KMeans(k=10, max_iterations=300)
train_labels = kmeans.fit(x_train)

#%%% Predict con x_test

cluster_labels = kmeans.predict(x_test)

#%%% sklearn (sólo como comparación)

from sklearn.cluster import KMeans as sKMeans

skmeans = sKMeans(n_clusters=10, random_state=3, n_init="auto").fit(x_test)
labels = skmeans.labels_



#%% GRÁFICAS

#%%% Etiquetas reales 

np.random.seed(3)

# Reducción de dimensionalidad con PCA
n_componentes = 80
pca = PCA(n_components=n_componentes)
x_test_reducido = pca.fit_transform(x_test)

# Seleccionar una muestra aleatoria de 1000 registros
indices_muestra = np.random.choice(x_test_reducido.shape[0], 1000, replace=False)

muestra = x_test_reducido[indices_muestra, :]

muestra_labels = y_test[indices_muestra]

# Graficar la muestra en función de los dos primeros componentes principales
plt.figure(figsize=(10, 8))

for cluster in range(10):  # 10 clusters
    indices_cluster = np.where(muestra_labels == cluster)
    plt.scatter(
        muestra[indices_cluster, 0],
        muestra[indices_cluster, 1],
        label=f'Cluster {cluster}',
        alpha=0.7
    )

plt.title('Muestra Aleatoria de 1000 Registros: Etiquetas Reales')
plt.xlabel('Primer Componente Principal')
plt.ylabel('Segundo Componente Principal')
plt.legend()
plt.show()

#%%% K-Means clustering

muestra_labels = cluster_labels[indices_muestra]

# Graficar la muestra en función de los dos primeros componentes principales
plt.figure(figsize=(10, 8))

for cluster in range(10):  # 10 clusters
    indices_cluster = np.where(muestra_labels == cluster)
    plt.scatter(
        muestra[indices_cluster, 0],
        muestra[indices_cluster, 1],
        label=f'Cluster {cluster}',
        alpha=0.7
    )

plt.title('Muestra Aleatoria de 1000 Registros: K-Means')
plt.xlabel('Primer Componente Principal')
plt.ylabel('Segundo Componente Principal')
plt.legend()
plt.show()

#%%% Sklearn (Sólo como prueba)

muestra_labels = labels[indices_muestra]

# Graficar la muestra en función de los dos primeros componentes principales
plt.figure(figsize=(10, 8))

for cluster in range(10):  # 10 clusters
    indices_cluster = np.where(muestra_labels == cluster)
    plt.scatter(
        muestra[indices_cluster, 0],
        muestra[indices_cluster, 1],
        label=f'Cluster {cluster}',
        alpha=0.7
    )

plt.title('Muestra Aleatoria de 1000 Registros: Sklearn')
plt.xlabel('Primer Componente Principal')
plt.ylabel('Segundo Componente Principal')
plt.legend()
plt.show()