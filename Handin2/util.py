import numpy as np


def get_data(file):
    train_file = np.load(file)
    images = train_file['digits']
    labels = train_file['labels']

    return images, labels
