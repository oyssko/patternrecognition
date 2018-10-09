import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_decision_boundary(pred_func, data, label):
    # Set min and max values and give it some padding
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap='Spectral', alpha=0.3)
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap='Spectral', edgecolors='k')


class NeuralNetwork:
    """
        NeuralNetwork: Class for creating and training a neural network

    """
    def __init__(self, layers, alpha=1, negpos = True):
        """
        Init will initialize the weights

        :param layers: An array that consists of information about the amount of nodes en each layer. Example:
        layer = [2, 5, 3, 2] => input layer has 2 nodes, hidden layer 1 has 5 nodes, hidden layer two has 3 nodes. The
        last number corresponds to the output layer, in this case is 2 nodes.
        :param alpha: Alpha value for the logistic activation function
        :param negpos:  If True, weights are initialized with random values in range [-1, 1]
        if False, weights are initialized with random values in range [0, 1]

        """

        # Random seed
        np.random.seed(0)
        self.layers = layers
        # initialize weights
        self.W, self.b = self.init_weights_and_biases(negpos)
        self.out = []
        self.alpha = alpha

        self.figer = plt.figure()
        self.axer = self.figer.add_subplot(111)

    def init_weights_and_biases(self, negpos):

        # Empty list
        W=[]
        b=[]
        if negpos:
            # Between -1 and 1
            for i in range(len(self.layers)):
                if i == len(self.layers)-1:
                    b.append(2 * np.random.random([self.layers[i]])-1)
                    break
                elif i == 0:
                    W.append(2*np.random.random([self.layers[i], self.layers[i+1]])-1)
                else:
                    W.append(2 * np.random.random([self.layers[i], self.layers[i + 1]]) - 1)
                    b.append(2*np.random.random([self.layers[i]])-1)
        else:
            # Between 0 and 1
            for i in range(len(self.layers)):
                if i == len(self.layers)-1:
                    b.append(np.random.random([self.layers[i]]))
                    break
                elif i == 0:
                    W.append(np.random.random([self.layers[i], self.layers[i+1]]))
                else:
                    W.append(np.random.random([self.layers[i], self.layers[i + 1]]))
                    b.append(np.random.random([self.layers[i]]))
        return W, b

    def sigmoid(self, x, a):
        # Activation function
        return 1/(1+np.exp(-x*a))

    def d_sigmoid(self, sig, a):
        # Derivative of activation function
        return a * sig * (1 - sig)

    def feedforward(self, data):
        # initialize an empty list for output values from each layer
        self.out = []
        for i in range(len(self.W)):
            if i == 0:
                self.out.append(self.sigmoid(np.dot(data, self.W[i]) + self.b[i], self.alpha))
            else:
                self.out.append(self.sigmoid(np.dot(self.out[i - 1], self.W[i]) + self.b[i],
                                             self.alpha))

        return self.out

    def prediction(self, data, conf=False):
        # Prediction function -> returns 0 for class 1 and 1 for class 2
        out = self.feedforward(data)

        self.predict = np.argmax(out[len(out)-1], axis=1)
        # Confidence of prediction:
        if conf:
            confidence = np.max(out[len(out)-1], axis=1) / np.sum(out[len(out)-1], axis=1)

            return self.predict, confidence
        else:
            return self.predict

    def accuracy(self, label):
        res = self.predict == np.argmax(label, axis=1)
        return np.sum(res) / len(res)

    def backpropagation(self, data, label, learning):
        """
            data = input array
            label = true label
        """
        networkOut = self.out[len(self.out) - 1]
        er = ((label - networkOut)**2).mean()
        out_delta = []
        layers = len(self.W)
        b_delta = []
        # Calculate deltas
        for i, j in zip(reversed(range(layers)), range(layers)):

            if i == layers - 1:
                out_delta.append((label - networkOut) * self.d_sigmoid(networkOut, self.alpha))
                b_delta.append(np.sum(out_delta[j], axis=0))
            else:
                out_delta.append(out_delta[j - 1].dot(self.W[i+1].T) * self.d_sigmoid(self.out[i],
                                                                                      self.alpha))
                b_delta.append(np.sum(out_delta[j], axis=0))

        # Update weights
        for i, j in zip(reversed(range(layers)), range(layers)):
            if i == 0:
                self.W[i] += data.T.dot(out_delta[j]) * learning
                self.b[i] += learning * b_delta[j]
            else:
                self.W[i] += self.out[i-1].T.dot(out_delta[j]) * learning
                self.b[i] += learning * b_delta[j]


        return er

    def training(self, data, label, learning, epochs=2000, marker=1000, breakpoint=0.0001):
        error = []
        bar = tqdm(range(epochs))
        for epoch in bar:
            idx = np.random.permutation(len(label))
            data, label = data[idx], label[idx]

            self.feedforward(data)
            er = self.backpropagation(data, label, learning)

            if epoch % marker == 0:
                bar.set_description("Error: %f" %er)
                bar.update()
                error.append(er)
            if er < breakpoint:
                bar.close()
                break


        print("\n Broke out of for loop with ", er, "Error")
        error = np.asarray(error, dtype=np.float64)
        plotx = range(len(error))
        plt.figure()
        plt.plot(plotx, error)
        plt.xlabel("every %i epoch" %marker)
        plt.ylabel("Error")
        plt.show()

