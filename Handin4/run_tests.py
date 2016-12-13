import sklearn.decomposition
import sklearn.datasets
from Handin4.EM import em, most_likely
from Handin4.evaluating_clusters import f1, silhouette
from sklearn.mixture import GaussianMixture

# Load Iris data set
iris = sklearn.datasets.load_iris()
data = iris['data']
labels = iris['target']
# Apply PCA
pca = sklearn.decomposition.PCA(2)
data_pca = pca.fit_transform(data)

# testing EM:
mean, cov, prior = em(data, 3, 0.1)
predicted = most_likely(data, mean, cov, prior)

print(f1(predicted, labels))