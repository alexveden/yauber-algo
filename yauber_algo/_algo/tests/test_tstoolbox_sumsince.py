import unittest
from yauber_algo.errors import *


class SumSinceTestCase(unittest.TestCase):
    def test_Sum_since(self):
        import yauber_algo.sanitychecks as sc

        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import sum_since

        #
        # Function settings
        #
        algo = 'sum_since'
        func = sum_since

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, 2, 5, 4, 9, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond
                    False,  # first_is_zero
                ),
                suffix='cond_increasing'
            )

            s.check_regular(
                array([nan, 2, 5, nan, 5, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, nan, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),    # cond
                    False,

                ),
                suffix='arr_nan_at_arr'
            )

            s.check_regular(
                array([nan, 2, 5, nan, 10, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 3, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, nan, 0, 1], dtype=np.float),  # cond
                    False,

                ),
                suffix='arr_nan_at_cond'
            )

            s.check_regular(
                array([nan, 0, 3, 0, 5, 0], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond
                    True,  # first_is_zero
                ),
                suffix='first_is_zero'
            )

            s.check_regular(
                array([nan, 0, 3, 0, 5, 0], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 2, 1], dtype=np.float),  # cond
                    True,  # first_is_zero
                ),
                suffix='wrong_condition',
                exception=YaUberAlgoInternalError,
            )

            s.check_naninf(
                array([nan, 2, nan, 4, 9, nan], dtype=np.float),
                func, (
                    array([inf, 2, 3, 4, 5, 6], dtype=np.float),  # arr
                    array([0, 1, nan, 1, 0, nan], dtype=np.float),  # cond
                    False,  # first_is_zero
                ),
                suffix='cond_is_nan_codex20180209'
            )

            s.check_naninf(
                array([nan, 2, nan, 4, 9, nan], dtype=np.float),
                func, (
                    array([inf, 2, nan, 4, 5, inf], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond
                    False,  # first_is_zero
                ),
                suffix='arr_is_nan_codex20180209'
            )

            s.check_series(
                pd.Series(array([nan, 2, 5, 4, 9, 6], dtype=np.float)),
                func, (
                    pd.Series(array([1, 2, 3, 4, 5, 6], dtype=np.float)),  # arr
                    pd.Series(array([0, 1, 0, 1, 0, 1], dtype=np.float)),  # cond
                    False,  # first_is_zero
                ),
                suffix=''
            )

            s.check_dtype_float(
                array([nan, 2, 5, 4, 9, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.float),  # cond
                    False,  # first_is_zero
                ),
                suffix='cond_increasing'
            )

            s.check_dtype_int(
                array([nan, 2, 5, 4, 9, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.int32),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.int32),  # cond
                    False,  # first_is_zero
                ),
                suffix='cond_increasing'
            )

            s.check_dtype_bool(
                array([nan, 2, 5, 4, 9, 6], dtype=np.float),
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.float),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.bool),  # cond
                    False,  # first_is_zero
                ),
                suffix='cond_increasing'
            )

            s.check_dtype_object(
                func, (
                    array([1, 2, 3, 4, 5, 6], dtype=np.object),  # arr
                    array([0, 1, 0, 1, 0, 1], dtype=np.bool),  # cond
                    False,  # first_is_zero
                ),
                suffix='cond_increasing'
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(100),
                               np.random.random_integers(0, 1, 100),
                               5
                           ),
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(100),
                                           np.random.random_integers(0, 1, 100),
                                           5
                                       ),
                                       )
