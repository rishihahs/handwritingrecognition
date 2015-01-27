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

    def cost(self, a, z):
        J = numpy.sum(-1 * self.Y * numpy.log(a[-1]))
        J = J + numpy.sum(-1 * (1 - self.Y) * numpy.log(1 - a[-1]))
        J = 1/self.X.shape[0] * J

        return J

    def backpropogation(self, a, z):
        delta = [a[-1] - self.Y]
        for i in range(len(z) - 2):
            dot = numpy.dot(delta[0], numpy.delete(self.Theta[-1 - i], 0, 1))
            delta.insert(0, dot * self.__sigmoidGradient(z[-2 - i]))

        Delta = []
        for i in range(len(delta)):
            shapeddelta = numpy.transpose(1/self.X.shape[0] * numpy.dot(numpy.transpose(a[i]), delta[i]))
            Delta.append(shapeddelta.ravel())

        return Delta

    def forwardpropogation(self, Theta):
        a = []
        z = [None] # First element is None, since z starts at layer 2

        a1 = self.X
        a1 = numpy.hstack((numpy.ones((a1.shape[0], 1), dtype=numpy.float64),  a1)) # Prepend columns with ones
        a.append(a1)

        for i in range(len(Theta)):
            zcurrent = numpy.dot(a[i], numpy.transpose(Theta[i]))
            z.append(zcurrent)
            acurrent = self.__sigmoid(zcurrent)

            if i < len(Theta) - 1:
                acurrent = numpy.hstack((numpy.ones((acurrent.shape[0], 1), dtype=acurrent.dtype),  acurrent)) # Prepend columns with ones

            a.append(acurrent)

        return (a, z)

    def __sigmoid(self, z):
        return 1 / (1 + numpy.exp(-1 * z))

    def __sigmoidGradient(self, z):
        g = self.__sigmoid(z)
        return g * (1 - g)

    def prints(self):
        print(self.__backpropogation())

if __name__ == '__main__':
    # stuff = data.loaddata(sys.argv[1])
    # neuralnetwork = NeuralNetwork([532, 53, 26], stuff.X_train, stuff.Y_train)
    # neuralnetwork.prints()
    neuralnetwork = NeuralNetwork([5, 3], numpy.array([[0.0312, 0.1392, 0.0246], [0.01342, 0.1322, 0.023456], [0.02943, 0.1632, 0.04654], [0.02333, 0.124352, 0.023432]]), numpy.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]))
    thetas = neuralnetwork.Theta
    thetavals = []
    for i in range(len(thetas)):
        thetavals.append(thetas[i].ravel())
    Theta = numpy.concatenate(thetavals)

    e = 1e-4
    gradapprox = []
    for p in range(len(Theta)):
        thetaplus = numpy.copy(Theta)
        thetaplus[p] = thetaplus[p] + e
        thetaminus = numpy.copy(Theta)
        thetaminus[p] = thetaminus[p] - e
        thetagradp = []
        thetagradm = []
        lastsize = 0
        print(thetaplus[p])
        print(thetaminus[p])
        for i in range(len(neuralnetwork.Theta)):
            t = neuralnetwork.Theta[i]
            thetagradp.append(thetaplus[lastsize:lastsize+t.size].reshape(t.shape))
            thetagradm.append(thetaminus[lastsize:lastsize+t.size].reshape(t.shape))
            lastsize = lastsize + t.size
        gradapprox.append((neuralnetwork.cost(*neuralnetwork.forwardpropogation(thetagradp)) - neuralnetwork.cost(*neuralnetwork.forwardpropogation(thetagradm))) / (2 * e))

    real = numpy.concatenate(neuralnetwork.backpropogation(*neuralnetwork.forwardpropogation(neuralnetwork.Theta)))

    for i in range(len(gradapprox)):
        print([gradapprox[i], real[i]])
