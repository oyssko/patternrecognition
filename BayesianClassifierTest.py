import numpy as np
from patternrecognition.classifiers import BayesianClassifier2d
from sklearn.datasets import make_spd_matrix

mean3d = [[1.0, 1.0, 1.0],
         [1.5, 1.5, 1.5]]
cov3d = [[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
         [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]]]

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

number_of_classes = 15
number_of_samples = 50
meanRand = np.random.randint(0, 20, size=(number_of_classes, 2))
vecRand = np.random.randint(0, 5, size=(50, 2))
covRand = []
for i in range(number_of_classes):
    covtemp = make_spd_matrix(2, random_state=10)
    covRand.append(covtemp)

mean3drand = np.random.randint(0, 10, size=(5, 3))
cov3drand = []
for i in range(5):
    covtemp = make_spd_matrix(3, random_state=10)
    cov3drand.append(covtemp)

bayes3D = BayesianClassifier2d(mean3d, cov3d, number_of_samples)
bayes3D.calculate_probability()
bayes3D.prediction_of_data()
bayes3D.accuracy()
bayes3D.visualize()

testBayes = BayesianClassifier2d(mean[0:2], cov[0:2], number_of_samples)
testBayes.calculate_probability()
testBayes.prediction_of_data()
testBayes.accuracy()
pred = testBayes.get_prediction(vecRand, display_pred=False)
print(pred)
testBayes.visualize()
