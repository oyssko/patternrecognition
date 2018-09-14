import numpy as np
from patternrecognition.classifiers import BayesianClassifier
from sklearn.datasets import make_sparse_spd_matrix
import scipy.stats


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
        [1.5, 1.5],
        [7, 3],
        [3, 5],
        [2, 5],
        [1, 3]]
cov = [[[0.2, 0], [0, 0.2]],
       [[0.2, 0], [0, 0.2]],
       [[0.2, 0], [0, 0.2]],
       [[0.3, 0], [0, 0.3]],
       [[0.6, 0], [0, 0.6]],
       [[0.5, 0], [0, 0.2]],
       [[0.34, 0], [0, 0.34]]]

number_of_classes = 5
number_of_samples = 200

meanRand = np.random.randint(0, 50, size=(number_of_classes, 2))
vecRand = np.random.randint(-2, 10, size=(50, 1)) * np.random.rand(50, 2)
covRand = np.zeros([number_of_classes, 2, 2])
for i in range(number_of_classes):
    covRand[i, :, :] = 10 * make_sparse_spd_matrix(dim=2, alpha=0.95, norm_diag=True,  random_state=45)


mean3drand = np.random.randint(0, 50, size=(number_of_classes, 3))
cov3drand = []

for i in range(number_of_classes):
    covtemp = 10 * make_sparse_spd_matrix(dim=3, alpha=0.95, norm_diag=False,  random_state=None)
    cov3drand.append(covtemp)

risk_mat = np.random.rand(number_of_classes, number_of_classes)
np.fill_diagonal(risk_mat, 0)


risk_mat2class = np.array([[0, 1], [0.5, 0]])
priorprin = [0.5, 0.5]

bayes3D = BayesianClassifier(mean3d[0:2], cov3d[0:2],prior_prob=priorprin, num_samples=number_of_samples, name='bayes3d', display=True, generate_data=True)
bayes3D.prediction_of_data(method='bayes')
bayes3D.accuracy()
bayes3D.visualize()

testBayes_distance = BayesianClassifier(mean[0:2], cov[0:2],prior_prob=priorprin, num_samples=number_of_samples, name='Minimize error probability',
                                        display=True)
testBayes_distance.prediction_of_data(method='bayes')
testBayes_distance.accuracy()
testBayes_distance.visualize()


testBayes_bayes = BayesianClassifier(mean[0:2], cov[0:2], prior_prob=priorprin, num_samples=number_of_samples, name='Mimimize average risk',
                                     risk_mat=risk_mat, display=True)
testBayes_bayes.prediction_of_data(method='bayes')
testBayes_bayes.accuracy()
testBayes_bayes.visualize()


