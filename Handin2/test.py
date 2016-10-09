import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


train_file = np.load('Data/auTrain.npz')
print(train_file.keys())  # ['digits', 'labels']
images = train_file['digits']
labels = train_file['labels']
print('Shape of input data: %s' % (images.shape,))
print('Shape of input labels: %s' % (labels.shape,))

test_file = np.load('Data/auTest.npz')
print(test_file.keys())
images_test = test_file['digits']
labels_test = test_file['labels']
print('Shape of test input data: %s' % (images_test.shape,))

clf = svm.SVC(kernel='linear',C=1)
clf.fit(images, labels)
print(clf.support_vectors_.shape)
plt.imshow(clf.support_vectors_[0].reshape(28,-1),cmap='bone')

in_sample_accuracy = (clf.predict(images)==labels).mean()
print('In sample Accuracy {:.2%}, In Sample Error {:.4f} '.format(in_sample_accuracy,1-in_sample_accuracy))
out_of_sample_accuracy = (clf.predict(images_test)==labels_test).mean()
print('Out of sample Accuracy {:.2%}, In Sample Error {:.4f}'.format(out_of_sample_accuracy,1-out_of_sample_accuracy))

plt.show()