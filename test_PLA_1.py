# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:24:13 2022

@author: jacksonj
"""

from PLA_1 import PLA_1
import numpy as np


def test_PLA1():

    # Load the test sets
    with np.load("PLA_1_test_sets.npz") as npz_file:

        # Obtain list of arrays
        file_names = npz_file.files
        file_name_iter = iter(file_names)

        # Check proposed solution against reference solution
        # for each test set
        for _ in range(5):
            X = npz_file[next(file_name_iter)]
            y = npz_file[next(file_name_iter)]
            w = PLA_1(X, y)
            w_ref = npz_file[next(file_name_iter)]
            print(w, w_ref)
            assert np.allclose(w, w_ref)


if __name__ == "__main__":
    test_PLA1()
