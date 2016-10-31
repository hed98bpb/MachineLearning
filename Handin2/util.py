import numpy as np


def get_data(file):
    train_file = np.load(file)
    images = train_file['digits']
    labels = train_file['labels']
    return images, labels

def get_data_nn(file):
    train_file = np.load(file)
    images = train_file['digits']
    labels = train_file['labels']

    n = images.shape[0]
    Y = np.zeros((n, 10))  # Y.shape = n x K
    for i in range(n):
        Y[i][labels[i]] = 1

    return images, Y