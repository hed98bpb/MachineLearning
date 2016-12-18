import numpy as np
from scipy.spatial import distance

def f1(predicted, labels):
    n, = predicted.shape
    assert labels.shape == (n,)
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

    # TODO: Implement the F1 score here
    contingency = np.zeros((r,k))
    for x in range(n):
        contingency[predicted[x]][labels[x]] += 1

    F_individual = np.zeros(r)
    for c in range(r):
        prec = np.max(contingency[c,:])/np.sum(contingency[c,:])
        recall= np.max(contingency[c,:])/np.sum(contingency[:, np.argmax(contingency[c,:])])

        F_individual[c]= (2* prec*recall) / (prec + recall)

    F_overall = np.sum(F_individual)/r

    assert contingency.shape == (r, k)
    return F_individual, F_overall, contingency

def silhouette(data, predicted):
    data = np.asarray(data)
    n, d = data.shape
    predicted = np.squeeze(np.asarray(predicted))
    k = np.max(predicted) + 1
    assert predicted.shape == (n,)

    # TODO: Implement the computation of the silhouette coefficient for each data point here.
    s = np.zeros(n)
    cluster_size = [1 for i in range(k)]

    for i in range(n):
        mean_in = 0
        sum_in = 0
        for j in range(n):
            if predicted[j] == predicted[i] and j != i:
                sum_in += distance.euclidean(data[i], data[j])
                cluster_size[predicted[i]] += 1
        mean_in = sum_in / (cluster_size[predicted[i]] - 1)

        mean_out = np.zeros(k)
        mean_out[predicted[i]] = np.inf
        for j in range(n):
                if predicted[i] != predicted[j]:
                    mean_out[predicted[j]] += distance.euclidean(data[i], data[j])

        for l in range(k):
            mean_out[l] = mean_out[l] / cluster_size[l]

        min_mean_out = np.min(mean_out)

        s[i] = (mean_in - min_mean_out) / max(mean_in, min_mean_out)


    assert s.shape == (n,)
    return s