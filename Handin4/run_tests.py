import sklearn.decomposition
import sklearn.datasets
from Handin4.EM import em, most_likely
from Handin4.evaluating_clusters import f1, silhouette
from Handin4.plot_handling import plot_groups
from sklearn.mixture import GaussianMixture
from Handin4.Image_compression import compress_facade, compress_stairs
import numpy as np
"""
# Load Iris data set
iris = sklearn.datasets.load_iris()
data = iris['data']
labels = iris['target']
# Apply PCA
pca = sklearn.decomposition.PCA(2)
data_pca = pca.fit_transform(data)

# testing EM on 2D-iris data:
print('Running EM on 2D-iris data')
mean, cov, prior = em(data_pca, 2, 0.001)
print('MEAN:')
print(mean)
print('PRIOR:')
print(prior)
print('COVARIANCE:')
print(cov)

predicted = most_likely(data_pca, mean, cov, prior)

# f1-score
F_indu, F_overall, cont = f1(predicted, labels)
print('CONTINGENCY:')
print(cont)
print('F-MEASURE:')
print(F_overall)

print('SILHUETTE:')
s = silhouette(data_pca, predicted)
print(s)
print('SILHUETTE COEFFICIENT:', (np.sum(s)/len(labels)))

print('PREDICTED:')
print(predicted)
print('LABELS')
print(labels)

# Plot
plot_groups(data_pca, predicted, {0: 'o', 1: 's', 2: '^'}, figsize=(4, 4))

"""

# playing with images:
compress_facade()

