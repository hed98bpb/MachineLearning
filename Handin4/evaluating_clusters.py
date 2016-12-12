import numpy as np

def f1(predicted, labels):
    n, = predicted.shape
    assert labels.shape == (n,)
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

    # TODO: Implement the F1 score here
    contingency = ...
    F_individual = ...
    F_overall = ...

    assert contingency.shape == (r, k)
    return F_individual, F_overall, contingency

def silhouette(data, predicted):
    data = np.asarray(data)
    n, d = data.shape
    predicted = np.squeeze(np.asarray(predicted))
    k = np.max(predicted) + 1
    assert predicted.shape == (n,)

    # TODO: Implement the computation of the silhouette coefficient for each data point here.
    s = ...

    assert s.shape == (n,)
    return s