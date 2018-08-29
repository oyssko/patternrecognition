import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis


class BayesianClassifier2d(object):
    def __init__(self, mean, cov, num_samples, risk_mat=None, display=True, name = None):
        self.mean = mean
        self.cov = cov
        self.num_samples = num_samples
        self.risk_mat = risk_mat
        self.name = name
        self.samples = []
        self.classes = self.mean.__len__()
        self.dim = len(self.mean[0][:])
        self.color = []
        # Generate colors
        for i in range(self.classes):
            color = list(np.random.choice(np.arange(0, 1, 0.01), size=3))
            self.color.append(color)

        if self.dim > 3:
            self.display = False
        else:
            self.display = display

        # Generate samples
        for num_class in range(self.classes):
            sample = self.generate_samples(self.mean[num_class], self.cov[num_class], self.num_samples)
            self.samples.append(sample)
        # Combine samples
        self.combined = np.concatenate(self.samples)

        # Mark the true classes
        self.trueClass = np.array_split(np.zeros([self.classes * self.num_samples]), self.classes)
        for num_class in range(self.classes):
            self.trueClass[num_class][:] = num_class
        self.trueClass = np.concatenate(self.trueClass)

    def display_true(self):
        # Display true classes
        self.fig_true = plt.figure()
        if self.dim == 3:
            self.z = []
            self.ax_true = self.fig_true.add_subplot(111, projection='3d')
        else:
            self.ax_true = self.fig_true.add_subplot(111)
        self.x = []
        self.y = []
        for num_class in range(self.classes):
            if self.dim == 3:
                xtemp, ytemp, ztemp = self.samples[num_class].T
                self.z.append(ztemp)
                self.x.append(xtemp)
                self.y.append(ytemp)
                self.ax_true.scatter(self.x[num_class], self.y[num_class], self.z[num_class], marker='*',
                                     c=self.color[num_class])
            else:
                xtemp, ytemp = self.samples[num_class].T
                self.x.append(xtemp)
                self.y.append(ytemp)
                self.ax_true.scatter(self.x[num_class], self.y[num_class], marker='*', c=tuple(self.color[num_class]))

    @staticmethod
    def generate_samples(meanVal, covariance, numberSam):
        # Function for generating samples of numberSam elements
        samples = np.random.multivariate_normal(meanVal, covariance, numberSam)
        return samples

    @staticmethod
    def distance(mean, vector, covariance, method='euclidean'):

        if method == 'euclidean':
            return np.linalg.norm(vector - mean)

        elif method == 'mahalanobis':
            cov_inv = np.linalg.inv(covariance)
            return mahalanobis(vector, mean, cov_inv)
        else:
            return print("Specify method or use default method (euclidean)")

    def gaussian_probability(self, x, meanClass, covariance):
        # Calculates probability of x given mean and covariance
        probability = multivariate_normal.pdf(x, mean=meanClass, cov=covariance)
        return probability

    def calculate_probability(self):
        # Retrieving the probability for belonging to each class for each vectot
        if self.risk_mat is None:
            self.probability = np.zeros([self.classes, self.classes * self.num_samples])
            for num_class in range(self.classes):
                self.probability[num_class, :] = self.gaussian_probability(self.combined,
                                                                           self.mean[num_class],
                                                                           self.cov[num_class])
        else:
            self.probability = np.zeros([self.classes, self.classes * self.num_samples])
            for i in range(self.classes):
                for j in range(self.classes):
                    self.probability[i, :] += self.risk_mat[i, j] * self.gaussian_probability(self.combined,
                                                                                              self.mean[i],
                                                                                              self.cov[i])

    def calculate_distance(self):
        # Calculating distance for between the classes and each vector
        self.vec_distance = np.zeros([self.classes, self.classes * self.num_samples])
        # if the covariances are equal use euclidean distance
        if all(np.array_equal(self.cov[0], x) for x in self.cov):
            for num_class in range(self.classes):
                for i in range(self.classes * self.num_samples):
                    self.vec_distance[num_class, i] = self.distance(self.mean[num_class],
                                                                    self.combined[i], self.cov[0],
                                                                    method='euclidean')
        # if the covariances are not equal use mahalanobis distance
        else:
            for num_class in range(self.classes):
                for i in range(self.classes * self.num_samples):
                    self.vec_distance[num_class, i] = self.distance(self.mean[num_class],
                                                                    self.combined[i], self.cov[num_class],
                                                                    method='mahalanobis')

    def prediction_of_data(self, method='bayes'):
        # This calculates the most probable class for each vector using Bayes Decision theory and distance
        # measure
        # Predicted class
        if method == 'bayes':
            self.calculate_probability()
            self.class_predict = self.probability.argmax(axis=0)
        elif method == 'distance':
            self.calculate_distance()
            self.class_predict = self.vec_distance.argmin(axis=0)
        else:
            print('Choose method as either  \'bayes\' or \'distance\'')
        #Display
        if self.display:
            self.fig_predict = plt.figure()
            if self.dim == 3:
                self.ax_predict = self.fig_predict.add_subplot(111, projection='3d')
            else:
                self.ax_predict = self.fig_predict.add_subplot(111)
            for num_class in range(self.classes):
                for i in range(self.class_predict.shape[0]):
                    if self.class_predict[i] == num_class:
                        if self.dim == 3:
                            self.ax_predict.scatter(self.combined[i][0], self.combined[i][1],
                                                    self.combined[i][2], marker='*', c=self.color[num_class])
                        else:
                            self.ax_predict.scatter(self.combined[i][0], self.combined[i][1],
                                                    marker='*', c=self.color[num_class])

    def get_prediction(self, vector, display_pred=True):
        # This method takes in vectors and predicts what class it belongs to
        prob = np.zeros([self.classes, len(vector)])
        for num_class in range(self.classes):
            prob[num_class, :] = self.gaussian_probability(vector, self.mean[num_class], self.cov[num_class])
        pred = prob.argmax(axis=0)
        # Display below
        if display_pred:
            if self.classes != 2:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            else:
                color = {0: 'r', 1: 'b'}
            for num_class in range(self.classes):
                if self.classes != 2:
                    color = list(np.random.choice(np.arange(0, 1, 0.01), size=3))
                for i in range(len(vector)):
                    if pred[i] == num_class:
                        if self.classes == 2:
                            self.ax_predict.scatter(vector[i][0], vector[i][1], marker='o', c=color[num_class])
                        else:
                            ax.scatter(vector[i][0], vector[i][1], marker='o', c=tuple(self.color[num_class]))
            max_val = np.amax(self.combined[::])
            min_val = np.amin(self.combined[::])
            if self.classes != 2:
                ax.set_ylim(min_val, max_val)
                ax.set_xlim(min_val, max_val)
                ax.set_title('Predictions from unknown data')
        return pred

    def accuracy(self):

        accurate = sum(self.class_predict == self.trueClass) / (self.classes * self.num_samples)
        print(self.name, ': The accuracy is %s' % (accurate,))

        return accurate

    def boundary(self):
        # Creating the separate boundaries between the PDFs
        x = np.arange(np.amin(self.combined[::]), np.amax(self.combined[::]), 0.1)
        y = np.arange(np.amin(self.combined[::]), np.amax(self.combined[::]), 0.1)
        self.xb, self.yb = np.meshgrid(x, y)
        pos = np.empty(self.xb.shape + (2,))
        pos[:, :, 0] = self.xb
        pos[:, :, 1] = self.yb
        Z1 = self.gaussian_probability(pos, self.mean[0], self.cov[0])
        Z2 = self.gaussian_probability(pos, self.mean[1], self.cov[1])
        # difference of Gaussians
        Z = (Z2 - Z1)
        return Z

    def visualize(self):
        # Display the true classes and the predicted classes
        if self.display:
            if (self.classes == 2) & (self.dim != 3):
                bound = self.boundary()
                self.ax_predict.contourf(self.xb, self.yb, bound, alpha=0.3, cmap='jet')

            self.display_true()
            max_val = np.amax(self.combined[::])
            min_val = np.amin(self.combined[::])
            self.ax_predict.set_ylim(min_val, max_val)
            self.ax_predict.set_xlim(min_val, max_val)
            self.ax_true.set_ylim(min_val, max_val)
            self.ax_true.set_xlim(min_val, max_val)
            self.ax_predict.set_title('Predictions from known data')
            self.ax_true.set_title('True distribution')
            plt.show()
        elif self.dim > 3:
            print('Can not visualize more than 3 dimensions ')
        else:
            print('Display is False')
