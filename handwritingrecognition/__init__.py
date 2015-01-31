from handwritingrecognition import neuralnetwork
from handwritingrecognition import data
import sys
import os
import numpy

def main():
    # Train Neural Network
    print('Loading data...')
    stuff = data.loaddata(sys.argv[1])

    Theta = neuralnetwork.randomtheta([63, 63, 26], stuff.X_train.shape[1])

    print('Training...')
    output = neuralnetwork.train(stuff.X_train, stuff.Y_train, Theta, 0.16, maxiterations = 1233)

    # Save correct Thetas
    print('Saving Thetas...')
    numpy.savez_compressed(os.path.abspath(sys.argv[2]), *output[0])
