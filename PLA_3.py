# -*- coding: utf-8 -*-
import numpy as np
from PLA_2 import PLA_2

SEED = 424242
rng = np.random.default_rng(SEED)


def PLA_rand(X, y):
    '''
    Implement more sophisticated PLA that randomly selects a misclassified
    example to use in the PLA update. An all-zero initial weight vector is 
    used. Compare performance of this PLA with simpler one that uses the
    first misclassifed example in updates.

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

    # Obtain N and d. Should be 50 and 3.
    (N, d) = X.shape

    # initialize weight vector with d+1 0's (w0, w1, ..., wd)
    w = np.zeros(d+1)

    # initialize count of number of PLA updates
    t = 0

    # prepend vector of N 1's as first column of X
    X1 = np.hstack((np.ones((N, 1)), X))

    # Continue as long as at least one example is misclassified
    done = False
    while not done:

        # Generate array of indices of all misclassified examples.
        # Note that np.nonzero() returns a tuple containing the
        # array we want as its first (and only) element.
        predicted_y = np.sign(X1 @ w)
        indices_of_misclassified = np.nonzero(predicted_y != y)[0]

        # Use a random misclassified example, if any, to perform a PLA
        # weight vector update. Otherwise, we are done updating.
        if indices_of_misclassified.size > 0:
            index_of_misclassified = rng.choice(indices_of_misclassified)
            w = w + X1[index_of_misclassified] * y[index_of_misclassified]
            t += 1
        else:
            done = True

    return (w, t)


def PLA_3(Xs, ys):
    '''
    Compare number of updates for PLA that selects
    a random misclassified example vs. the version that always uses the
    first misclassified example.

    Parameters
    ----------
    Xs : NumPy 50-by-3-by-100 array
        100 50-by-3 X test data sets.
    ys : NumPy 50-by-100 array
        100 50-element y labels corresponding to X data sets.

    Returns
    -------
    NumPy 3-element vector
        First element is percentage of time random-misclassified 
        algorithm is faster.
        Second/third element is average number of PLA updates for
        random-/first-misclassified algorithm.

    '''

    # number of times random algorithm is faster
    n_rand_wins = 0
    # total number of PLA updates by random-misclassified algorithm
    n_rand_its = 0
    # total number of PLA updates by first-misclassified algorithm
    n_first_its = 0
    # number of times to run and compare the two algorithms
    runs = 100

    for r in range(runs):
        X = Xs[:, :, r]
        y = ys[:, r]
        (_, t_rand) = PLA_rand(X, y)
        (_, t_first) = PLA_2(X, y)
        if t_rand < t_first:
            n_rand_wins += 1
        n_rand_its += t_rand
        n_first_its += t_first

    # Run pytest with -rP option to see output generated by the following
    print(f"Percentage random wins: {n_rand_wins/runs}")
    print(f"Average random iterations: {n_rand_its/runs}")
    print(f"Average first-misclassified iterations: {n_first_its/runs}")

    return np.array([n_rand_wins, n_rand_its, n_first_its])/runs
