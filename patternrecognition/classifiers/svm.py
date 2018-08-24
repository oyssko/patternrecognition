import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

"""
TODO: Add more functionality in branching:
    - Kernel implementation
    - More feature for each classifier
    - More SVM methods:
        - Gaussian
        - RBF
        - SVC with polynomial 
        etc.
"""
style.use('ggplot')


class Support_Vector_Machine(object):
    def __init__(self, vizualization=True):
        self.vizualization = vizualization
        self.colors = {1: 'r', -1: 'b'}
        if self.vizualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data

        # { ||w||: [w, b]}
        opt_dict = {}

        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]
        all_data = []

        for yi in self.data:
            for featureset in data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.maxFeatureVal = float('-inf')
        self.minFeatureVal = float('inf')

        for yi in self.data:
            if np.amax(self.data[yi]) > self.maxFeatureVal:
                self.maxFeatureVal = np.amax(self.data[yi])
            if np.amin(self.data[yi]) < self.minFeatureVal:
                self.minFeatureVal = np.amin(self.data[yi])

        all_data = None

        step_sizes = [self.maxFeatureVal * 0.1,
                      self.maxFeatureVal * 0.01,
                      self.maxFeatureVal * 0.001]
        b_range_multiple = 3
        b_multiple = 5

        latest_optimum = self.maxFeatureVal * 10

        for steps in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.maxFeatureVal * b_range_multiple),
                                   self.maxFeatureVal * b_range_multiple,
                                   steps * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if yi * (np.dot(w_t, xi) + b) < 1:
                                    found_option = False
                        if found_option:
                            #TODO: FIX THIS ISSUE WITH opt_dict
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print('optimized a step.')
                else:
                    w = w - steps
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + steps * 2

    def predict(self, features):
        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.vizualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in self.data[i]] for i in self.data]

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        data_range = (self.minFeatureVal * 0.9, self.maxFeatureVal * 1.1)
        hyp_xmin = data_range[0]
        hyp_xmax = data_range[1]

        psv1 = hyperplane(hyp_xmin, self.w, self.b, 1)
        psv2 = hyperplane(hyp_xmax, self.w, self.b, 1)
        self.ax.plot([hyp_xmin, hyp_xmax], [psv1, psv2])

        nsv1 = hyperplane(hyp_xmin, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_xmax, self.w, self.b, -1)
        self.ax.plot([hyp_xmin, hyp_xmax], [nsv1, nsv2])

        db1 = hyperplane(hyp_xmin, self.w, self.b, 0)
        db2 = hyperplane(hyp_xmax, self.w, self.b, 0)
        self.ax.plot([hyp_xmin, hyp_xmax], [db1, db2])

        plt.show()
