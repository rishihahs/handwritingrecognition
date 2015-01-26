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
        thetacolumns = max(X.shape[1], max(layers)) # Maximum of all layer units
        epsilon = math.sqrt(6) / math.sqrt(layers[-2] + layers[-1]) # Statistically good epsilon
        self.epsilon = epsilon
        self.Theta = (2 * epsilon) * numpy.random.random_sample((len(layers), thetacolumns)) - epsilon # Randomized Theta

if __name__ == '__main__':
    stuff = data.loaddata(sys.argv[1])
    neuralnetwork = NeuralNetwork([532, 53, 26], stuff.X_train, stuff.Y_train)
