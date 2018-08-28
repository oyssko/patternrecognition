import numpy as np
import matplotlib.pyplot as plt
from mpltools import layout
from scipy.stats import multivariate_normal


class BayesianClassifier2d:
    def __init__(self, mean, cov, num_samples, display=True):
        self.mean = mean
        self.cov = cov
        self.num_samples = num_samples
        self.samples = []
        self.classes = self.mean.__len__()
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

        figsize = layout.figaspect(aspect_ratio=1)
        self.fig_true, self.ax_true = plt.subplots(ncols=1, figsize=figsize)
        self.x = []
        self.y = []
        for num_class in range(self.classes):
            color = list(np.random.choice(np.arange(0, 1, 0.01), size=3))
            xtemp, ytemp = self.samples[num_class].T
            self.x.append(xtemp)
            self.y.append(ytemp)
            self.ax_true.plot(self.x[num_class], self.y[num_class], '*', c=tuple(color))

    @staticmethod
    def generate_samples(meanVal, covariance, numberSam):
        # Function for generating samples of numberSam elements
        samples = np.random.multivariate_normal(meanVal, covariance, numberSam)
        return samples

    def distance(self, mean, feature, method='euclidean'):

        if method == 'euclidean':
            return np.linalg.norm(feature - mean)
        else:
            return print("Specify method or use default method (euclidean)")

    def gaussian_probability(self, x, meanClass, covariance):
        # Calculates probability of x given mean and covariance
        probability = multivariate_normal.pdf(x, mean=meanClass, cov=covariance)
        return probability

    def calculate_probability(self):
        # Retrieving the probability for each point
        self.probability = np.zeros([self.classes, self.classes * self.num_samples])
        for num_class in range(self.classes):
            for vector in range(self.classes * self.num_samples):
                self.probability[num_class, vector] = self.gaussian_probability(self.combined[vector],
                                                                                self.mean[num_class],
                                                                                self.cov[num_class])

    def predict_data(self):

        self.figure_predict = plt.figure()
        self.ax_predict = self.figure_predict.add_subplot(111)
        self.class_predict = self.probability.argmax(axis=0)

        if self.display:
            for num_class in range(self.classes):
                color = list(np.random.choice(np.arange(0, 1, 0.01), size=3))
                for i in range(self.class_predict.shape[0]):
                    if self.class_predict[i] == num_class:
                        self.ax_predict.scatter(self.combined[i][0], self.combined[i][1], marker='*', c=tuple(color))

    def accuracy(self):

        accurate = sum(self.class_predict == self.trueClass) / (self.classes * self.num_samples)
        print('The accuracy is %s' % (accurate,))

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
            if self.classes == 2:
                bound = self.boundary()
                self.ax_predict.contourf(self.xb, self.yb, bound, alpha=0.3, cmap='jet')
            self.display_true()
            max_val = np.amax(self.combined[::])
            min_val = np.amin(self.combined[::])
            self.ax_predict.set_ylim(min_val, max_val)
            self.ax_predict.set_xlim(min_val, max_val)
            self.ax_true.set_ylim(min_val, max_val)
            self.ax_true.set_xlim(min_val, max_val)
            self.ax_predict.set_title('Predictions')
            self.ax_true.set_title('True distribution')
            plt.show()
