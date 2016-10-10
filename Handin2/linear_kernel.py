from sklearn import svm

from Handin2.util import get_data

images, labels = get_data('Data/auTrain.npz')
images_test, labels_test = get_data('Data/auTest.npz')

for i in range(10):
    print('C:', 0.1*(i+1))

    clf = svm.SVC(kernel='linear',C=(0.1*(i+1)))
    clf.fit(images, labels)

    in_sample_accuracy = (clf.predict(images)==labels).mean()
    print('In sample Accuracy {:.2%}, In Sample Error {:.4f} '.format(in_sample_accuracy,1-in_sample_accuracy))

    out_of_sample_accuracy = (clf.predict(images_test)==labels_test).mean()
    print('Out of sample Accuracy {:.2%}, In Sample Error {:.4f}'.format(out_of_sample_accuracy,1-out_of_sample_accuracy))
