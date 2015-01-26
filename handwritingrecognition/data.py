import numpy
import random
import os
import sys
import math
import string
import collections

X_train, X_cv, X_test = None, None, None
Y_train, Y_cv, Y_test = None, None, None

def loaddata(directory):
    global X_train, X_cv, X_test, Y_train, Y_cv, Y_test

    if not X_train:
        X, y = __extract_data(directory)

        # 60% of data is for training
        mainlen, otherlen = math.floor(0.6 * len(X)), math.ceil(0.2 * len(X))

        X_train, X_cv, X_test = X[0:mainlen], X[mainlen:mainlen + otherlen], X[mainlen + otherlen:]

        Y = __multify(y)
        Y_train, Y_cv, Y_test = Y[0:mainlen], Y[mainlen:mainlen + otherlen], Y[mainlen + otherlen:]

    return collections.namedtuple('Data', 'X_train X_cv X_test Y_train Y_cv Y_test')(X_train, X_cv, X_test, Y_train, Y_cv, Y_test)

def __multify(y):
    ynew = numpy.zeros((len(y), 26)) # 26 letters of the alphabet
    alphabet = string.ascii_lowercase

    for i in range(len(y)):
        index = alphabet.find(y[i])
        ynew[i, index] = 1
    return ynew

def __extract_data(directory):
    # Load data into example vector
    data = numpy.load(os.path.abspath(directory))

    keys = data.keys()
    keys.sort() # Super slow if not sorted

    # Create m x n array
    X = numpy.empty((len(data[keys[0]]) * len(keys), len(data[keys[0]][0])), dtype=numpy.float64)
    y = numpy.empty(len(X), dtype=numpy.object)

    coordinates = set() # store used coordinates to avoid collisions

    for key in keys:
        for example in data[key]:
            # Choose random row
            row = numpy.random.randint(0, len(X))
            while row in coordinates:
                row = (row + 1) % (len(X) - 1)
            coordinates.add(row)

            y[row] = key

            # Copy over columns
            for j in range(len(example) - 1):
                X[row, j] = 2 * example[j] / 255 - 1 # Multiply by 2/255 - 1 to get the intensity value

    data.close()

    return (X, y)

if __name__ == '__main__':
    data = loaddata(sys.argv[1])
    print(len(data.X_train))
    print(len(data.X_cv))
    print(len(data.X_test))
    print(Y_train[0])
