# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:24:13 2022

@author: jacksonj
"""

from PLA_3 import PLA_3
import numpy as np


def test_PLA3():

    # Load the test sets
    with np.load("PLA_3_test_sets.npz") as npz_file:

        # Obtain list of arrays
        file_names = npz_file.files
        file_name_iter = iter(file_names)

        # Check proposed solution against reference solution
        # for each test set
        for _ in range(5):
            Xs = npz_file[next(file_name_iter)]
            ys = npz_file[next(file_name_iter)]
            results = PLA_3(Xs, ys)
            results_ref = npz_file[next(file_name_iter)]
            assert np.allclose(results, results_ref)


if __name__ == "__main__":
    test_PLA3()
