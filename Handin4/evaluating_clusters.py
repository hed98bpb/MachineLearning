import numpy as np
from scipy.spatial import distance

def f1(predicted, labels):
    n, = predicted.shape
    assert labels.shape == (n,)
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

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

    s = np.zeros(n)
    cluster_size = [1 for i in range(k)]

    for i in range(n):
        sum_in = 0
        for j in range(n):
            if i!=j:
                cluster_size[predicted[j]] += 1
                if predicted[j] == predicted[i] and j != i:
                    sum_in += distance.euclidean(data[i], data[j])

        mean_in = sum_in / (cluster_size[predicted[i]] - 1)

        mean_out = np.zeros(k)
        mean_out[predicted[i]] = np.inf
        for j in range(n):
                if predicted[i] != predicted[j]:
                    mean_out[predicted[j]] += distance.euclidean(data[i], data[j])



        min_mean_out = np.zeros(k)
        for l in range(k):
            min_mean_out[l] = mean_out[l]/cluster_size[l]

        s[i] = (np.min(min_mean_out) - mean_in) / max(mean_in, np.min(min_mean_out))

    assert s.shape == (n,)
    return s