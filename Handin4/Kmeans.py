import numpy as np
from scipy.spatial import distance
import random


def closest(data, centers):
    n, d = data.shape
    k, d_ = centers.shape
    assert d == d_
    rep = np.zeros(n, dtype=np.int)

    # Our code for finding the center that is closest to each entry in data
    for i in range(n):
        min_dist = (np.inf, 0)
        for c in range(k):
            # calculating euclidean distance
            dist = distance.euclidean(data[i],centers[c])
            if dist < min_dist[0]:
                min_dist = (dist, c)

        rep[i] = int(min_dist[1])

    # rep should contain a representative index for each data point
    assert rep.shape == (n,)
    assert np.all((0 <= rep) & (rep < k))
    return rep


def kmeans_cost(data, centers, rep):
    n, d = data.shape
    k, d_ = centers.shape
    assert d == d_
    assert rep.shape == (n,)

    data_rep = centers[rep]

    TD_C = np.zeros(k)
    for i in range(n):
        dist = distance.euclidean(data[i], data_rep[i])
        TD_C[rep[i]] += dist**2
    TD_C = np.sqrt(TD_C)

    cost = np.sqrt(np.sum(TD_C**2))

    return cost


def kmeans(data, k, epsilon):
    data = np.asarray(data)
    n, d = data.shape

    # Initialize centers with random chosen points in data
    k_random_nbs = random.sample(range(n), k)
    centers = data[k_random_nbs, :]

    tired = False
    old_centers = np.zeros_like(centers)
    while not tired:
        print('not tired')
        old_centers[:] = centers

        # Reassign points to nearest center
        rep = closest(data, centers)

        sum = np.zeros((k, d))
        size = np.zeros(k)

        for i in range(n):
            size[rep[i]] += 1
            for a in range(d):
                sum[rep[i]][a] += data[i][a]

        for i in range(k):
            if size[i] == 0:
                # Handling empty clusters.
                print('FUCK! empty cluster in k-means - trying again')
                return kmeans(data, k, epsilon)
            else:
                centers[i :] = sum[i :] / size[i]

        dist = np.sqrt(((centers - old_centers) ** 2).sum(axis=1))
        tired = np.max(dist) <= epsilon

        cost = kmeans_cost(data, centers, rep)


    return centers, cost


