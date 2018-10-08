import numpy as np
import matplotlib.pyplot as plt

# Seed the random function to ensure that we always get the same result

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
    plt.contourf(xx, yy, Z, cmap='Spectral')
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap='Spectral')


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
        self.W = self.init_weights(negpos)
        self.out = []
        self.alpha = alpha

        self.figer = plt.figure()
        self.axer = self.figer.add_subplot(111)

    def init_weights(self, negpos):

        # Empty list
        W=[]
        if negpos:
            # Between -1 and 1
            for i in range(len(self.layers)):
                if i == len(self.layers)-1:
                    break
                W.append(2*np.random.random([self.layers[i], self.layers[i+1]])-1)
        else:
            # Between 0 and 1
            for i in range(len(self.layers)):
                if i == len(self.layers)-1:
                    break
                W.append(np.random.random([self.layers[i], self.layers[i+1]]))
        return W

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
                self.out.append(self.sigmoid(np.dot(data, self.W[i]), self.alpha))
            else:
                self.out.append(self.sigmoid(np.dot(self.out[i - 1], self.W[i]), self.alpha))

        return self.out

    def prediction(self, data, conf=False):

        out = self.feedforward(data)

        self.predict = np.argmax(out[len(out)-1], axis=1)

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
            Out = list of outputs
            data = input array
            label = true label
        """
        networkOut = self.out[len(self.out) - 1]
        er = abs(label - networkOut).mean()
        out_delta = []
        layers = len(self.W)

        for i, j in zip(reversed(range(layers)), range(layers)):

            if i == layers - 1:
                out_delta.append((label - networkOut) * self.d_sigmoid(networkOut, self.alpha))
            else:
                out_delta.append(out_delta[j - 1].dot(self.W[i+1].T) * self.d_sigmoid(self.out[i],
                                                                                      self.alpha))

        for i, j in zip(reversed(range(layers)), range(layers)):
            if i == 0:
                self.W[i] += data.T.dot(out_delta[j]) * learning
            else:
                self.W[i] += self.out[i-1].T.dot(out_delta[j]) * learning

        return er

    def training(self, data, label, learning, epochs=2000, marker=1000):
        error = []
        plotx = range(int(epochs/marker))
        for epoch in range(epochs):
            idx = np.random.permutation(len(label))
            data, label = data[idx], label[idx]

            self.feedforward(data)
            er = self.backpropagation(data, label, learning)

            if epoch % marker == 0:
                print('Error after ', epoch,"epochs: ", er)
                error.append(er)

        error = np.asarray(error, dtype=np.float64)
        plt.figure()
        plt.plot(plotx, error)
        plt.xlabel("every %i epoch" %marker)
        plt.ylabel("Error")
        plt.show()
