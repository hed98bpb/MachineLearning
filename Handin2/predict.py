import numpy as np

from sklearn import svm

from Handin2.util import get_data

images, labels = get_data('Data/auTrain.npz')
images_test, labels_test = get_data('Data/auTest.npz')


def predict(testdata):

    clf = svm.SVC(kernel='rbf', C=100, gamma=0.01)
    clf.fit(images, labels)

    res = clf.predict(testdata)

    assert len(res) == testdata.shape[0]

    return res

