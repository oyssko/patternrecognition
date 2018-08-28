import numpy as np
from patternrecognition.classifiers import BayesianClassifier2d
from sklearn.datasets import make_spd_matrix

mean = [[1.0, 1.0],
        [1.5, 1.5],
        [5, 3],
        [7, 3],
        [3, 1],
        [4, 5],
        [1, 3]]
cov = [[[0.2, 0], [0, 0.2]],
       [[0.2, 0], [0, 0.2]],
       [[0.2, 0], [0, 0.2]],
       [[0.3, 0], [0, 0.3]],
       [[0.6, 0], [0, 0.6]],
       [[0.5, 0], [0, 0.2]],
       [[0.34, 0], [0, 0.34]]]

number_of_classes = 30
number_of_samples = 1000
meanRand = np.random.randint(0, 30, size=(number_of_classes, 2))
covRand = []
for i in range(number_of_classes):
    covtemp = make_spd_matrix(2, random_state=None)
    covRand.append(covtemp)

testBayes = BayesianClassifier2d(mean[0:2], covRand[0:2], number_of_samples, display=True)
testBayes.calculate_probability()
testBayes.predict_data()
testBayes.accuracy()
testBayes.visualize()
