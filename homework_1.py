import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.spatial import distance
import imageio.v2 as imageio
import os

iris = load_iris()
X = iris.data[:, :2]

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Метод локтя (Scikit-learn)')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()

optimal_k = 3

def initialize_centroids(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    clusters = np.zeros(len(X), dtype=int)
    for i, x in enumerate(X):
        distances = [distance.euclidean(x, centroid) for centroid in centroids]
        clusters[i] = np.argmin(distances)
    return clusters

def update_centroids(X, clusters, k):
    centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
    return centroids

def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    old_centroids = np.zeros_like(centroids)
    clusters = np.zeros(len(X), dtype=int)

    images = []
    os.makedirs('kmeans_images', exist_ok=True)

    for iteration in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)

        plt.figure()
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        for i in range(k):
            plt.scatter(X[clusters == i, 0], X[clusters == i, 1], color=colors[i % len(colors)], label=f'Кластер {i+1}')
        plt.scatter(new_centroids[:, 0], new_centroids[:, 1], marker='x', s=200, linewidths=3, color='black', label='Центроиды')
        plt.title(f'Итерация {iteration + 1}')
        plt.legend()

        filename = f'kmeans_images/iteration_{iteration+1}.png'
        plt.savefig(filename)
        plt.close()
        images.append(imageio.imread(filename))

        if np.allclose(centroids, new_centroids):
            print(f"Алгоритм сошелся за {iteration + 1} итераций.")
            break

        centroids = new_centroids
        old_centroids = centroids.copy()

    imageio.mimsave('kmeans_animation.gif', images, duration=1)
    print("GIF-анимация создана: kmeans_animation.gif")


kmeans(X, optimal_k)
