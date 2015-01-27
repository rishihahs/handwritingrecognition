import numpy
from scipy import optimize
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

        # Forward propogation memoization
        self.memoizeforward = {}

    def train(self):
        # output = optimize.check_grad((lambda x: self.__cost(*self.__forwardpropogation(self.__reroll(x, self.Theta)))),
        #                             (lambda x: self.__backpropogation(*self.__forwardpropogation(self.__reroll(x, self.Theta)))),
        #                             self.__unroll(self.Theta))
        output = optimize.fmin_l_bfgs_b((lambda x: self.__cost(*self.__forwardpropogation(self.X, self.__reroll(x, self.Theta)))),
                                        self.__unroll(self.Theta),
                                        (lambda x: self.__backpropogation(*self.__forwardpropogation(self.X, self.__reroll(x, self.Theta)))),
                                        maxiter = 600)

        return output

    def __reroll(self, unrolled, target):
        arr = []
        pos = 0
        for element in target:
            arr.append(unrolled[pos:pos+element.size].reshape(element.shape))
            pos += element.size
        return arr

    def __unroll(self, matarray):
        newcpy = []
        for m in matarray:
            newcpy.append(m.ravel())
        return numpy.concatenate(newcpy)

    def __cost(self, a, z):
        J = numpy.sum(-1 * self.Y * numpy.log(a[-1]))
        J = J + numpy.sum(-1 * (1 - self.Y) * numpy.log(1 - a[-1]))
        J = 1/self.X.shape[0] * J

        print(J)

        return J

    def __backpropogation(self, a, z):
        delta = [a[-1] - self.Y]
        for i in range(len(z) - 2):
            dot = numpy.dot(delta[0], numpy.delete(self.Theta[-1 - i], 0, 1))
            delta.insert(0, dot * self.__sigmoidGradient(z[-2 - i]))

        Delta = []
        for i in range(len(delta)):
            shapeddelta = numpy.transpose(1/self.X.shape[0] * numpy.dot(numpy.transpose(a[i]), delta[i]))
            Delta.append(shapeddelta.ravel())

        return numpy.concatenate(Delta)

    def __forwardpropogation(self, X, Theta):
        hashkey = hash(str(Theta))
        if hashkey in self.memoizeforward:
            val = self.memoizeforward[hashkey]
            del self.memoizeforward[hashkey]
            return val

        a = []
        z = [None] # First element is None, since z starts at layer 2

        a1 = X
        a1 = numpy.hstack((numpy.ones((a1.shape[0], 1), dtype=numpy.float64),  a1)) # Prepend columns with ones
        a.append(a1)

        for i in range(len(Theta)):
            zcurrent = numpy.dot(a[i], numpy.transpose(Theta[i]))
            z.append(zcurrent)
            acurrent = self.__sigmoid(zcurrent)

            if i < len(Theta) - 1:
                acurrent = numpy.hstack((numpy.ones((acurrent.shape[0], 1), dtype=acurrent.dtype),  acurrent)) # Prepend columns with ones

            a.append(acurrent)

        retval = (a, z)
        self.memoizeforward[hashkey] = retval

        return retval

    def __sigmoid(self, z):
        return 1 / (1 + numpy.exp(-1 * z))

    def __sigmoidGradient(self, z):
        g = self.__sigmoid(z)
        return g * (1 - g)

if __name__ == '__main__':
    stuff = data.loaddata(sys.argv[1])
    print('Loaded Data')
    neuralnetwork = NeuralNetwork([63, 36, 26], stuff.X_train, stuff.Y_train)
    # neuralnetwork = NeuralNetwork([5, 3], numpy.array([[0.0312, 0.1392, 0.0246], [0.01342, 0.1322, 0.023456], [0.02943, 0.1632, 0.04654], [0.02333, 0.124352, 0.023432]]), numpy.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]))
    output = neuralnetwork.train()

    print('Testing')
