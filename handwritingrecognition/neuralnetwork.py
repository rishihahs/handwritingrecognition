import numpy
import math
import sys

import data

class NeuralNetwork(object):

    def __init__(self, layers, X, Y): # layers = [1 2 3] where the elements are number of units
        self.layers = layers
        self.X = X
        self.Y = Y

        # Create Theta
        self.Theta = []
        layers.insert(0, X.shape[1])

        for i in range(len(layers) - 1):
            epsilon = math.sqrt(6) / math.sqrt(layers[i] + layers[i + 1]) # Statistically good epsilon
            self.Theta.append((2 * epsilon) * numpy.random.random_sample((layers[i + 1], layers[i] + 1)) - epsilon) # Randomized Theta

    def __forwardpropogation(self):
        a = []
        z = [None] # First element is None, since z starts at layer 2

        a1 = self.X
        a1 = numpy.hstack((numpy.ones((a1.shape[0], 1), dtype=a1.dtype),  a1)) # Prepend columns with ones
        a.append(a1)

        for i in range(len(self.Theta)):
            zcurrent = numpy.dot(a[i], numpy.transpose(self.Theta[i]))
            z.append(zcurrent)
            acurrent = self.__sigmoid(zcurrent)
            acurrent = numpy.hstack((numpy.ones((acurrent.shape[0], 1), dtype=acurrent.dtype),  acurrent)) # Prepend columns with ones
            a.append(acurrent)

        return (a, z)

    def __sigmoid(self, z):
        return 1 / (1 + numpy.exp(-1 * z))

    def prints(self):
        print(self.__forwardpropogation())

if __name__ == '__main__':
    stuff = data.loaddata(sys.argv[1])
    neuralnetwork = NeuralNetwork([532, 53, 26], stuff.X_train, stuff.Y_train)
    neuralnetwork.prints()
