import unittest
from numpy import array, nan, inf
import os
import sys
import pandas as pd
import numpy as np
from yauber_algo.errors import *
from yauber_algo._algo.featurewiz import insert_sorted_inplace
from numba import jit


@jit
def is_sorted(a):
    for i in range(a.size-1):
        if a[i+1] < a[i]:
            return False
    return True

class FeatureWizTestCase(unittest.TestCase):
    def test_insert_sorted_inplace(self):
        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 0, 4)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([0, 1, 2, 3], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 2, 2)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, 3, 4], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, nan], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 2, 2)
        self.assertEqual(1, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, 3, nan], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 1.5, 2)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 1.5, 3, 4], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 2.5, 2)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2.5, 3, 4], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 2.5, 3)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, 2.5, 4], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 3.5, 3)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, 3.5, 4], dtype=np.float), equal_nan=True))


        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 2, 4)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, 2, 3], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 4, 2)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 3, 4, 4], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 0, 4)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([0, 1, 2, 3], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 5, 4)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, 3, 5], dtype=np.float), equal_nan=True))



        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, nan, 4)
        self.assertEqual(1, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, 3, nan], dtype=np.float), equal_nan=True))

        # Nan
        a = np.array([1, 2, nan, nan], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 4, nan)
        self.assertEqual(1, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, 4, nan], dtype=np.float), equal_nan=True))

        # Nan
        a = np.array([1, 2, nan, nan], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 0, nan)
        self.assertEqual(1, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([0, 1, 2, nan], dtype=np.float), equal_nan=True))



        # Nan
        a = np.array([1, 2, nan, nan], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, nan, nan)
        self.assertEqual(2, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, nan, nan], dtype=np.float), equal_nan=True))

        # inf
        a = np.array([1, 2, nan, nan], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, inf, inf)
        self.assertEqual(2, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, nan, nan], dtype=np.float), equal_nan=True))

        # inf
        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 5, 1)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([2, 3, 4, 5], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 1.5, 2)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 1.5, 3, 4], dtype=np.float), equal_nan=True))

        a = np.array([1, 2, 3, 4], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 2.5, 2)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2.5, 3, 4], dtype=np.float), equal_nan=True))

        #
        # Add nan
        # Window roll
        a = np.array([nan, nan, nan, nan], dtype=np.float)
        nan_cnt = insert_sorted_inplace(a, 1, nan)
        self.assertEqual(3, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, nan, nan, nan], dtype=np.float), equal_nan=True))

        nan_cnt = insert_sorted_inplace(a, 2, nan)
        self.assertEqual(2, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, nan, nan], dtype=np.float), equal_nan=True))

        nan_cnt = insert_sorted_inplace(a, 3, nan)
        self.assertEqual(1, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, 3, nan], dtype=np.float), equal_nan=True))

        nan_cnt = insert_sorted_inplace(a, 4, nan)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([1, 2, 3, 4], dtype=np.float), equal_nan=True))

        nan_cnt = insert_sorted_inplace(a, 5, 1)
        self.assertEqual(0, nan_cnt)
        self.assertEqual(True, np.allclose(a, np.array([2, 3, 4, 5], dtype=np.float), equal_nan=True))


        #
        # Check that results are always sorted!
        #
        for i in range(100000):
            arr = np.sort(np.random.random(10))
            a = arr.copy()
            new, _ = np.random.random(2)
            # Old must be always one of the element of array
            old = a[np.random.random_integers(0, len(a)-1)]
            nan_cnt = insert_sorted_inplace(a, new, old)
            self.assertEqual(0, nan_cnt)
            if not is_sorted(a):
                nan_cnt = insert_sorted_inplace(arr.copy(), new, old)
                self.assertEqual(True, False)