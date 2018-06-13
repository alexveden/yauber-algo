import unittest
from yauber_algo.errors import *


class MaxTestCase(unittest.TestCase):
    def test_max(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import max

        #
        # Function settings
        #
        algo = 'max'
        func = max

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([3, 2, 2, 4, 3, 2, 1], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 0, 1], dtype=np.float),
                    array([3, 2, 2, 1, 1, 2, 1], dtype=np.float),
                ),
                suffix='regular'
            )

            s.check_regular(
                array([3, 2, 2, 4, 3, 2, 1], dtype=np.float),
                func,
                (
                    array([3, 2, 2, 1, 1, 2, 1], dtype=np.float),
                    array([3, 2, 1, 4, 3, 0, 1], dtype=np.float),
                ),
                suffix='regular_reverse'
            )

            s.check_naninf(
                array([nan, 2, nan, 4, 3, 2, nan], dtype=np.float),
                func,
                (
                    array([inf, 2, nan, 1, 1, 2, nan], dtype=np.float),
                    array([3, 2, 1, 4, 3, 0, 1], dtype=np.float),
                ),
                suffix='nan_inf_arr1'
            )

            s.check_naninf(
                array([nan, 2, nan, 4, 3, 2, nan], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 0, 1], dtype=np.float),
                    array([inf, 2, nan, 1, 1, 2, nan], dtype=np.float),

                ),
                suffix='nan_inf_arr2'
            )

            s.check_series(
                pd.Series(array([3, 2, 2, 4, 3, 2, 1], dtype=np.float)),
                func,
                (
                    pd.Series(array([3, 2, 1, 4, 3, 0, 1], dtype=np.float)),
                    pd.Series(array([3, 2, 2, 1, 1, 2, 1], dtype=np.float)),
                ),
                suffix=''
            )

            s.check_dtype_float(
                array([3, 2, 2, 4, 3, 2, 1], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 0, 1], dtype=np.float),
                    array([3, 2, 2, 1, 1, 2, 1], dtype=np.float),
                ),
                suffix='regular'
            )

            s.check_dtype_int(
                array([3, 2, 2, 4, 3, 2, 1], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 0, 1], dtype=np.int32),
                    array([3, 2, 2, 1, 1, 2, 1], dtype=np.int32),
                ),
                suffix='regular'
            )

            s.check_dtype_bool(
                array([0, 1, 1, 1, 0, 1, 1], dtype=np.float),
                func,
                (
                    array([0, 0, 1, 0, 0, 0, 1], dtype=np.bool),
                    array([0, 1, 0, 1, 0, 1, 1], dtype=np.bool),
                ),
                suffix='regular'
            )

            s.check_dtype_object(
                func,
                (
                    array([0, 0, 1, 0, 0, 0, 1], dtype=np.object),
                    array([0, 1, 0, 1, 0, 1, 1], dtype=np.object),
                ),
                suffix='regular'
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(100),
                               np.random.random(100),
                           ),
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(100),
                                           np.random.random(100),
                                       ),
                                       )




