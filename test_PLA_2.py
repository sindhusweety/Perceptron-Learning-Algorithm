# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:24:13 2022

@author: jacksonj
"""

from PLA_2 import PLA_2
import numpy as np


def test_PLA2():

    # Load the test sets
    with np.load("PLA_2_test_sets.npz") as npz_file:

        # Obtain list of arrays
        file_names = npz_file.files
        file_name_iter = iter(file_names)

        # Check proposed solution against reference solution
        # for each test set
        for _ in range(5):
            X = npz_file[next(file_name_iter)]
            y = npz_file[next(file_name_iter)]
            (w, t) = PLA_2(X, y)
            w_ref = npz_file[next(file_name_iter)]
            t_ref = npz_file[next(file_name_iter)][0]
            assert np.allclose(w, w_ref)
            assert t == t_ref


if __name__ == "__main__":
    test_PLA2()
