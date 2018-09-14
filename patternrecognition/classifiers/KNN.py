import numpy as np

def k_nearest_neighbor(k, test_data, train_data):
    pred = np.zeros([len(test_data)])

    def distance(a, b):
        return np.linalg.norm(a - b)

    def smallestK_indices(a, Kth):
        idx = a.ravel().argsort()[:Kth]
        return np.stack(np.unravel_index(idx, a.shape)).T

    numclasses = len(train_data[:, 0, 0])
    for test in range(len(test_data)):
        dist = np.zeros([numclasses, len(train_data[0, :, 0])])
        classidx = np.zeros([numclasses])
        for classes in range(numclasses):
            for train in range(len(train_data[0, :, 0])):
                dist[classes, train] = distance(test_data[test, :], train_data[classes, train, :])

        distkminidx = smallestK_indices(dist, k)[:, 0]
        for classes in range(numclasses):
            classidx[classes] = np.sum(distkminidx == classes)

        pred[test] = classidx.argmax()

    return pred
