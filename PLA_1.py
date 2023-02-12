# -*- coding: utf-8 -*-

# TODO: SAVE YOUR SOLUTION AS PLA_1.py

import numpy as np



def PLA_1(X, y):
    '''
    Implement simple version of PLA on given training examples.
    This version uses an all-zero initial weight vector and
    uses the first misclassified example in the weight vector update.

    Parameters
    ----------
    X : 100 x 2 NumPy array
        100 points in two-dimensional space.
    y : 100-element NumPy vector
        +1/-1 labels for each of the X points.

    Returns
    -------
    w : 3-element NumPy vector
        Weight vector defining a hyperplane separating the data.
        w[0] is the bias value.

    '''

    # initialize weight vector w0, w1, w2 to all 0's
    w = np.zeros(3)


    # TODO: YOUR CODE GOES BELOW THIS LINE
    X = np.insert(X, obj=0, values= 1, axis=1) #axis 0 -> rows 1 -> col & obj -> index

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





    return w





# Extremely simple test case demonstrating how to call the function.
# Run this file to execute the test case below.
# Run test_PLA_1.py to execute test cases that will
# be used for grading.
if __name__ == "__main__":
    X = np.ones((100, 2))
    X[0][0] = -1
    X[0][1] = -1
    y = np.ones(100)
    y[0] = -1
    #X = np.array([[1, 1], [3, 1], [-1, 1]])
    #y = np.array([-1, 1, -1])
    print(PLA_1(X, y))  # Should be [-1 1 1]
