import numpy as np

from Handin2.predict import predict

from Handin2.util import get_data

images_test, labels_test = get_data('Data/auTest.npz')

def evaluate_student_classifier():
    ten_predictions = predict(images_test[0:13])
    correct = labels_test[0:13]
    assert ten_predictions.shape == correct.shape
    accuracy = np.mean(ten_predictions == correct)
    print("The student gets {:.2%} points".format(accuracy))

evaluate_student_classifier()