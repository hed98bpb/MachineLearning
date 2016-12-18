import sklearn.decomposition
import sklearn.datasets
from Handin4.EM import em, most_likely
from Handin4.evaluating_clusters import f1, silhouette
from sklearn.mixture import GaussianMixture
import numpy as np

# Load Iris data set
iris = sklearn.datasets.load_iris()
data = iris['data']
labels = iris['target']
# Apply PCA
pca = sklearn.decomposition.PCA(2)
data_pca = pca.fit_transform(data)

# testing EM on 2D-iris data:
print('Running EM on 2D-iris data')
mean, cov, prior = em(data_pca, 3, 0.001)
print('MEAN:')
print(mean)
print('PRIOR:')
print(prior)
print('COVARIANCE:')
print(cov)

predicted = most_likely(data_pca, mean, cov, prior)

print('F1:')
F_indu, F_overall, cont = f1(predicted, labels)
print(cont)
print(F_overall)

print('SILHUETTE:')
s = silhouette(data_pca, predicted)
print(s)
print('SILHUETTE COEFFICIENT:', (np.sum(s)/len(labels)))

print(predicted)
print(labels)