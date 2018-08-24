from patternrecognition.classifiers import Support_Vector_Machine
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
"""
Test code for patternrecognition.classifiers.Support_Vector_Machine
"""
(X, y) = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2, random_state=None)
X1 = np.c_[np.ones((X.shape[0])), X]

positive_X = []
negative_X = []

for i,v in enumerate(y):
    if v==0:
        negative_X.append(X[i])
    else:
        positive_X.append(X[i])

data_dict = {-1:np.array(negative_X), 1:np.array(positive_X)}

clg = Support_Vector_Machine()

clg.fit(data=data_dict)

prediction_data = [[1, 1],
                   [2, 6],
                   [3, 5],
                   [-1, -5],
                   [-4, 8],
                   [2, -9]]

for p in prediction_data:
    clg.predict(p)

clg.visualize()
