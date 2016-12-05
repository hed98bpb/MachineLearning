import numpy as np
import sklearn.datasets
import sklearn.decomposition
from scipy.spatial import distance
import random


# Load Iris data set
iris = sklearn.datasets.load_iris()
data = iris['data']
labels = iris['target']
# Apply PCA
pca = sklearn.decomposition.PCA(2)
data_pca = pca.fit_transform(data)


def closest(data, centers):
    n, d = data.shape
    k, d_ = centers.shape
    assert d == d_
    rep = np.zeros(n)

    # Our code for finding the center that is closest to each entry in data
    for i in range(n):
        min_dist = (np.inf, 0)
        for c in range(k):
            # calculating euclidean distance
            dist = distance.euclidean(data[i],centers[c])
            if dist < min_dist[0]:
                min_dist = (dist, c)

        rep[i] = min_dist[1]

    # rep should contain a representative index for each data point
    print(rep)
    assert rep.shape == (n,)
    assert np.all((0 <= rep) & (rep < k))
    return rep


def kmeans_cost(data, centers, rep):
    n, d = data.shape
    k, d_ = centers.shape
    assert d == d_
    assert rep.shape == (n,)

    # Insert your code here
    data_rep = centers[rep]
    cost = ...

    return cost


def kmeans(data, k, epsilon):
    data = np.asarray(data)
    n, d = data.shape

    # Initialize centers with random chosen points in data
    k_random_nbs = random.sample(range(n), k)
    centers = data[[k_random_nbs], :]

    tired = False
    old_centers = np.zeros_like(centers)
    while not tired:
        old_centers[:] = centers

        # Reassign points to nearest center
        # and handle empty clusters.
        #TODO

        dist = np.sqrt(((centers - old_centers) ** 2).sum(axis=1))
        tired = np.max(dist) <= epsilon

    return centers

random_nbs = random.sample(range(len(data)), 3)
print('randomly choosen numbers = entries for the random center-points in data: ', random_nbs)
closest(data, data[random_nbs, :])

