import numpy
from scipy import optimize
import math
import sys

import data

__memoizeforward = {} # Memoize forwardpropogation

def randomtheta(layers, num_features):
    # Create Theta
    Theta = []
    layers = list(layers)
    layers.insert(0, num_features)

    for i in range(len(layers) - 1):
        epsilon = math.sqrt(6) / math.sqrt(layers[i] + layers[i + 1]) # Statistically good epsilon
        Theta.append((2 * epsilon) * numpy.random.random_sample((layers[i + 1], layers[i] + 1)) - epsilon) # Randomized Theta

    return Theta

def train(X, Y, Theta, lambda_regularization, maxiterations = 600):
    output = optimize.fmin_l_bfgs_b((lambda x: __min_cost(x, X, Y, Theta, lambda_regularization)),
                                    __unroll(Theta),
                                    (lambda x: __min_gradient(x, X, Y, Theta, lambda_regularization)),
                                    maxiter = maxiterations)
    return output

def __min_cost(x, X, Y, Theta, lambda_regularization):
    rerolled = __reroll(x, Theta)
    return __cost(X, Y, rerolled, lambda_regularization, *__forwardpropogation(X, rerolled))

def __min_gradient(x, X, Y, Theta, lambda_regularization):
    rerolled = __reroll(x, Theta)
    return __backpropogation(X, Y, rerolled, lambda_regularization, *__forwardpropogation(X, rerolled))

def __reroll(unrolled, target):
    arr = []
    pos = 0
    for element in target:
        arr.append(unrolled[pos:pos+element.size].reshape(element.shape))
        pos += element.size
    return arr

def __unroll(matarray):
    newcpy = []
    for m in matarray:
        newcpy.append(m.ravel())
    return numpy.concatenate(newcpy)

def calculatecost(X, Y, Theta, targettheta, lambda_regularization):
    rerolled = __reroll(Theta, targettheta)
    return __cost(X, Y, rerolled, lambda_regularization, *__forwardpropogation(X, rerolled))

def __cost(X, Y, Theta, lambda_regularization, a, z):
    J = numpy.sum(-1 * Y * numpy.log(a[-1]))
    J = J + numpy.sum(-1 * (1 - Y) * numpy.log(1 - a[-1]))
    J = 1/X.shape[0] * J

    # Regularization
    reg = 0
    for theta in Theta:
        reg += numpy.sum(theta[0:,1:].ravel() ** 2)

    J = J + (lambda_regularization / (2 * X.shape[0])) * reg

    print(J)

    return J

def __backpropogation(X, Y, Theta, lambda_regularization, a, z):
    delta = [a[-1] - Y]
    for i in range(len(z) - 2):
        dot = numpy.dot(delta[0], numpy.delete(Theta[-1 - i], 0, 1))
        delta.insert(0, dot * __sigmoidGradient(z[-2 - i]))

    Delta = []
    for i in range(len(delta)):
        shapeddelta = numpy.transpose(1/X.shape[0] * numpy.dot(numpy.transpose(a[i]), delta[i]))

        # Regularization
        thetamod = numpy.copy(Theta[i])
        thetamod[:, 0] = 0
        shapeddelta = shapeddelta + (lambda_regularization / X.shape[0]) * thetamod

        Delta.append(shapeddelta.ravel())

    return numpy.concatenate(Delta)

def __forwardpropogation(X, Theta):
    global __memoizeforward

    hashkey = hash(str(Theta))
    if hashkey in __memoizeforward:
        val = __memoizeforward[hashkey]
        del __memoizeforward[hashkey]
        return val

    a = []
    z = [None] # First element is None, since z starts at layer 2

    a1 = X
    a1 = numpy.hstack((numpy.ones((a1.shape[0], 1), dtype=numpy.float64),  a1)) # Prepend columns with ones
    a.append(a1)

    for i in range(len(Theta)):
        zcurrent = numpy.dot(a[i], numpy.transpose(Theta[i]))
        z.append(zcurrent)
        acurrent = __sigmoid(zcurrent)

        if i < len(Theta) - 1:
            acurrent = numpy.hstack((numpy.ones((acurrent.shape[0], 1), dtype=acurrent.dtype),  acurrent)) # Prepend columns with ones

        a.append(acurrent)

    retval = (a, z)
    __memoizeforward[hashkey] = retval

    return retval

def __sigmoid(z):
    # Piecewise function to handle two extremes of exponential
    return numpy.piecewise(z,
                           [z >= 0, z < 0],
                           [(lambda x: 1 / (1 + numpy.exp(-1 * x))),
                            (lambda x: numpy.exp(x) / (1 + numpy.exp(x)))])

def __sigmoidGradient(z):
    g = __sigmoid(z)
    return g * (1 - g)

if __name__ == '__main__':
    stuff = data.loaddata(sys.argv[1])
    print('Loaded Data')
    # neuralnetwork = NeuralNetwork([63, 36, 26], stuff.X_train, stuff.Y_train)
    # X = numpy.array([[0.0312, 0.1392, 0.0246], [0.01342, 0.1322, 0.023456], [0.02943, 0.1632, 0.04654], [0.02333, 0.124352, 0.023432]])
    # Y = numpy.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    # Theta = randomtheta([5, 3], X.shape[1])
    # lambda_regularization = 0.2
    # output = optimize.check_grad((lambda x: __min_cost(x, X, Y, Theta, lambda_regularization)),
    #                             (lambda x: __min_gradient(x, X, Y, Theta, lambda_regularization)),
    #                             __unroll(Theta))

    # print(output)
