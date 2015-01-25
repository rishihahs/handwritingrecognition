import numpy

import random
import os
import sys

def _extract_data(directory):
    # Load data into example vector
    data = numpy.load(os.path.abspath(directory))

    keys = data.keys()
    keys.sort() # Super slow if not sorted

    # Create m x n array
    X = numpy.empty((len(data[keys[0]]) * len(keys), len(data[keys[0]][0])), dtype=numpy.uint8)
    y = numpy.empty(len(X), dtype=numpy.object)

    coordinates = set() # store used coordinates to avoid collisions

    for key in keys:
        for example in data[key]:
            # Choose random row
            row = random.randint(0, len(X) - 1)
            while row in coordinates:
                row = (row + 1) % (len(X) - 1)
            coordinates.add(row)

            y[row] = key

            # Copy over columns
            for j in range(len(example) - 1):
                X[row][j] = example[j]

    data.close()

    return (X, y)

if __name__ == '__main__':
    (X, y) = _extract_data(sys.argv[1])
