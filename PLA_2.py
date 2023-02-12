# -*- coding: utf-8 -*-

# TODO: SAVE YOUR SOLUTION AS PLA_2.py

import numpy as np


def PLA_2(X, y):
    '''
    Implement simple version of PLA on given training examples.
    This version uses an all-zero initial weight vector and
    uses the first misclassified example in the weight vector update.

    Parameters
    ----------
    X : N x d NumPy array
        N points in d-dimensional space.
    y : N-element NumPy vector
        +1/-1 labels for each of the X points.

    Returns
    -------
    w : (d+1)-element NumPy vector
        Weight vector defining a hyperplane separating the data.
        w[0] is the bias value.
    t : non-negative integer
        Number of PLA updates performed.

    '''

    # Obtain N and d.
    # TODO: REPLACE FOLLOWING LINE WITH CORRECT CODE.
    d = len(X[0])

    # initialize weight vector with d+1 0's (w0, w1, ..., wd)
    w = np.zeros(d+1)

    # initialize count of number of PLA updates
    t = 0

    # TODO: YOUR CODE GOES BELOW THIS LINE
    X = np.insert(X, obj=0, values=1, axis=1)

    while True:
        misclassified_index_point = None
        for INDEX, x in enumerate(X):
            if np.sign(np.dot(w.T, x)) != y[INDEX]:
                misclassified_index_point = INDEX
                break
        if misclassified_index_point is None:
            break
        else:
            w = w + y[misclassified_index_point] * X[misclassified_index_point]
            t += 1

    return (w, t)


# Extremely simple test case that should catch any constants erroneously
# remaining in the code copied from the part 1 solution.
# Run this file to execute the test case below.
# Run test_PLA_2.py to execute test cases that will
# be used for grading.
if __name__ == "__main__":
    X = np.ones((50, 3))
    X[0][0] = -1
    X[0][1] = -1
    X[0][2] = -1
    y = np.ones(50)
    y[0] = -1
    print(PLA_2(X, y))  # Should be ([-1 1 1 1], 1)
