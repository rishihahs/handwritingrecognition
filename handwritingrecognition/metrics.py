import matplotlib.pyplot
import numpy

import neuralnetwork
import data
import sys

def learningcurves(layers, X_train, Y_train, X_cv, Y_cv, lambda_regularization):
    trainplotx = []
    trainploty = []
    cvplotx = []
    cvploty = []

    trainlen = len(X_train)
    for i in reversed(range(30)):
        Theta = neuralnetwork.randomtheta(layers, X_train.shape[1])
        output = neuralnetwork.train(X_train[(trainlen - trainlen/(i + 1)):],
                               Y_train[(trainlen - trainlen/(i + 1)):],
                                     Theta,
                                     lambda_regularization,
                                     maxiterations = 233)
        trainplotx.append(trainlen/(i + 1))
        trainploty.append(output[1])

        cost = neuralnetwork.calculatecost(X_cv, Y_cv, output[0], Theta, lambda_regularization)
        cvplotx.append(trainlen/(i + 1))
        cvploty.append(cost)

    print(trainplotx)
    print(trainploty)
    print(cvplotx)
    print(cvploty)
    matplotlib.pyplot.plot(trainplotx, trainploty)
    matplotlib.pyplot.plot(cvplotx, cvploty)
    matplotlib.pyplot.show()

def lambda_check(layers, X_train, Y_train, X_cv, Y_cv):
    lambdas = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]
    costs = []
    for lambda_regularization in lambdas:
        Theta = neuralnetwork.randomtheta(layers, X_train.shape[1])
        output = neuralnetwork.train(X_train,
                                     Y_train,
                                     Theta,
                                     lambda_regularization,
                                     maxiterations = 233)

        cost = neuralnetwork.calculatecost(X_cv, Y_cv, output[0], Theta, 0)
        costs.append(cost)

    print(costs)
    print()
    print("Min Cost Is:")
    print(lambdas[numpy.argmin(costs)])
    matplotlib.pyplot.plot(lambdas, costs)
    matplotlib.pyplot.show()

if __name__ == '__main__':
    stuff = data.loaddata(sys.argv[1])
    # learningcurves([63, 63, 26], stuff.X_train, stuff.Y_train, stuff.X_cv, stuff.Y_cv, 0.16)
    lambda_check([63, 63, 26], stuff.X_train, stuff.Y_train, stuff.X_cv, stuff.Y_cv)
