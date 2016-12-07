import sklearn.decomposition
import sklearn.datasets
from Handin4.EM import em

# Load Iris data set
iris = sklearn.datasets.load_iris()
data = iris['data']
labels = iris['target']
# Apply PCA
pca = sklearn.decomposition.PCA(2)
data_pca = pca.fit_transform(data)


em(data, 3, 0.1)