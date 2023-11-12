"""
Created on Tue Nov  7 22:36:14 2023

@author: Bren Guzmán
"""

import numpy as np

class KMeans(object):
    def __init__(self, k, max_iterations=100, random_seed=None):
        self.k = k
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def initialize_centroids(self, data):
        indices = np.random.choice(data.shape[0], self.k, replace=False)
        centroids = data[indices, :]
        return centroids

    def assign_to_clusters(self, data, centroids):
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, data, labels):
        centroids = np.array([np.mean(data[labels == i, :], axis=0) for i in range(self.k)])
        return centroids

    def fit(self, data):
        centroids = self.initialize_centroids(data)

        for iteration in range(self.max_iterations):
            labels = self.assign_to_clusters(data, centroids)
            new_centroids = self.update_centroids(data, labels)

            # Verificar convergencia
            if np.all(centroids == new_centroids):
                print(f"Convergencia alcanzada en la iteración {iteration + 1}.")
                break

            centroids = new_centroids

        else:
            print(f"Se alcanzó el número máximo de iteraciones ({self.max_iterations}) sin converger.")

        return labels

