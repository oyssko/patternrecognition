import numpy as np
from patternrecognition.classifiers import BayesianClassifier2d
from sklearn.datasets import make_spd_matrix

mean3d = [[1.0, 1.0, 1.0],
          [1.5, 1.5, 1.5],
          [2.0, 2.0, 2.0],
          [3.0, 3.0, 3.0]]
cov3d = [[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
         [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
         [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
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
       [[0.5, 0], [0, 0.1]],
       [[0.3, 0], [0, 0.3]],
       [[0.6, 0], [0, 0.6]],
       [[0.5, 0], [0, 0.2]],
       [[0.34, 0], [0, 0.34]]]

number_of_classes = 15
number_of_samples = 100

meanRand = np.random.randint(0, 20, size=(number_of_classes, 2))
vecRand = np.random.randint(0, 5, size=(50, 2))
covRand = np.zeros([number_of_classes, 2, 2])
for i in range(number_of_classes):
    covRand[i, :, :] = make_spd_matrix(2, random_state=10)


mean3drand = np.random.randint(0, 2, size=(number_of_classes, 3))
cov3drand = []
for i in range(number_of_classes):
    covtemp = make_spd_matrix(3, random_state=10)
    cov3drand.append(covtemp)

risk_mat = np.random.rand(number_of_classes, number_of_classes)
np.fill_diagonal(risk_mat, 0)

risk_mat2d = np.array([[0, 1], [0.7, 0]])

bayes3D = BayesianClassifier2d(mean3d, cov3d, number_of_samples, name='bayes3d with risk')
bayes3D.calculate_probability()
bayes3D.calculate_distance()
bayes3D.prediction_of_data(method='distance')
bayes3D.accuracy()
bayes3D.visualize()


testBayes_distance = BayesianClassifier2d(meanRand, covRand, number_of_samples, name='distance classifier')
testBayes_distance.calculate_distance()
testBayes_distance.prediction_of_data(method='distance')
testBayes_distance.accuracy()
testBayes_distance.visualize()

testBayes_bayes = BayesianClassifier2d(meanRand, covRand, number_of_samples, risk_mat=risk_mat,
                                       name='bayes decision with risk')
testBayes_bayes.calculate_probability()
testBayes_bayes.prediction_of_data(method='bayes')
testBayes_bayes.accuracy()
testBayes_bayes.visualize()

