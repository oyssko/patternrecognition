import numpy as np


def kernel(input, typeofkern='normal'):
    def normal(u):
        factor = 1 / ((2 * np.pi) ** (1 / 2))

        return factor * np.exp(-(u ** 2) / 2)

    def uniform(u):
        pass

    if typeofkern == 'normal':
        return normal(input)
    elif typeofkern == 'uniform':
        return uniform(input)


class PdfEstimation(object):

    def __init__(self, training, testing):
        self.testing = testing
        self.training = training
        self.Ntr = len(training)
        self.Nte = len(testing)

    def parzen_window(self, h, method='normal'):
        Pest = np.zeros([self.Nte])

        for i in range(self.Nte):
            PestTemp = np.zeros([self.Ntr])
            for j in range(self.Ntr):
                x = self.testing[i]
                xi = self.training[j]
                PestTemp[j] = kernel(((x - xi) / h), typeofkern=method)

            Pest[i] = 1 / (h * self.Ntr) * np.sum(PestTemp)

        return Pest
