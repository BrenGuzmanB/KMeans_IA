"""
Created on Tue Nov  7 22:41:16 2023

@author: Bren Guzmán
"""
#%% LIBRERÍAS

from Dataframe import Dataframe
from KMeans import KMeans

#%% CARGAR LOS DATOS

# Ruta de los archivos
training_images_filepath = 'MNIST/t10k-images.idx3-ubyte'
training_labels_filepath = 'MNIST/t10k-labels.idx1-ubyte'

# Crear una instancia de la clase Dataframe
train_dataframe = Dataframe(training_images_filepath, training_labels_filepath)

# Cargar los datos
x_train, y_train = train_dataframe.load_data()

#%% K-Means


kmeans = KMeans(k=10, max_iterations=300, random_seed=3)
cluster_labels = kmeans.fit(x_train)

#%% sklearn (comparación)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=3, n_init="auto").fit(x_train)
labels = kmeans.labels_

#%%
import numpy as np

np.unique(y_train)
np.unique(y_train, return_counts=True)


np.unique(cluster_labels)
np.unique(cluster_labels, return_counts=True)

#%%
