import sklearn.decomposition
import sklearn.datasets
from Handin4.EM import em
from sklearn.mixture import GaussianMixture

# Load Iris data set
iris = sklearn.datasets.load_iris()
data = iris['data']
labels = iris['target']
# Apply PCA
pca = sklearn.decomposition.PCA(2)
data_pca = pca.fit_transform(data)


mean, cov, prior = em(data, 3, 0.1)

print('mean: ', mean, '\nprior: ', prior)