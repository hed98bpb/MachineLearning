from scipy.stats import multivariate_normal
import numpy as np
from Handin4.Kmeans import kmeans, closest


# HELPERFUNCTION:
# takes a description of a Gaussian Mixture (that is, the mean, covariance matrix and prior of each Gaussian)
# returns the probability densities of each point: P(x) slide 34 d. 23/11
def pdf(points, mean, cov, prior):
    points, mean, cov = np.asarray(points), np.asarray(mean), np.asarray(cov)
    prior = np.asarray(prior)
    n, d = points.shape
    k, d_1 = mean.shape
    k_2, d_2, d_3 = cov.shape
    k_3, = prior.shape
    assert d == d_1 == d_2 == d_3
    assert k == k_2 == k_3, "%s %s %s should be equal" % (k, k_2, k_3)

    # Compute probabilities
    prob = []
    for i in range(k):
        if prior[i] < 1 / k ** 3:
            prob.append(np.zeros(n))
        else:
            prob.append(
                prior[i] *
                multivariate_normal.pdf(
                    mean=mean[i], cov=cov[i], x=points))
    prob = np.transpose(prob)  # n x k
    # Normalize cluster probabilities of each point
    prob = prob / np.sum(prob, axis=1, keepdims=True)  # n x k

    assert prob.shape == (n, k)
    assert np.allclose(prob.sum(axis=1), 1)
    return prob

# HELPERFUNCTION:
# computes the most likely class of each point under a given Gaussian Mixture: P(C_i|x)
def most_likely(points, mean, cov, prior):
    prob = pdf(points, mean, cov, prior)
    return np.argmax(prob, axis=1)

# computing em
def em(points, k, epsilon, mean=None):
    points = np.asarray(points)
    n, d = points.shape

    # Initialize and validate mean
    if mean is None:
        # picks k points as mean
        min_cost = np.inf
        for i in range(10):
            centers, cost = kmeans(points, k, 0.01) # don't use the same epsilon in k-means as in EM.
            if cost < min_cost:
                mean = centers

    # Validate input
    mean = np.asarray(mean) # my_i in slide 36 d. 23/11
    k_, d_ = mean.shape
    assert k == k_
    assert d == d_

    # Initialize cov, prior (almost as done in http://www.vlfeat.org/overview/gmm.html)
    cov = np.zeros((k, d, d)) # eta_i in slide 36 d. 23/11
    prior = np.zeros(k) # W_i in slide 36 d. 23/11
    assigned_points = closest(points, mean)

    for i in range(k):
        cluster_id = (assigned_points==i)
        cluster = points[cluster_id, :]
        cluster_size = len(cluster)

        prior[i] = cluster_size / k

        # Handling empty clusters
        if cluster_size == 0:
            print('cluster empty in EM - starting over')
            return em(points, k, epsilon)
        else:
            cov[i,:,:] = np.cov(np.transpose(cluster))

    tired = False
    old_mean = np.zeros_like(mean)
    while not tired:
        old_mean[:] = mean

        # Expectation step:
        pdf_ = pdf(points, mean, cov, prior)

        # Maximization step:

            # re-assign prior
        for i in range(k):
            pdf_csum = np.sum(pdf_[:,i])
            prior[i] = 1/n * pdf_csum
            if prior[i] < 0.01: # handling almost empty cluster
                print('cluster empty in EM - starting over')
                return em(points, k, epsilon)


            # re-assign mean
            x_sum = 0
            for x in range(n):
                x_sum += x * pdf_[x][i]
            mean[i] = x_sum / pdf_csum

            # re-assign cov!
            cov_sum = 0
            for x in range(n):
                cov_sum += pdf_[x][i] * ((points[x] - mean[i]) * np.reshape(points[x] - mean[i], (d,1))) # reshape = transpose
            cov[i,:,:] = cov_sum / pdf_csum

        # Finish condition
        dist = np.sqrt(((mean - old_mean) ** 2).sum(axis=1))
        tired = np.all(dist < epsilon)

    # Validate output
    assert mean.shape == (k, d)
    assert cov.shape == (k, d, d)
    assert prior.shape == (k,)
    return mean, cov, prior

