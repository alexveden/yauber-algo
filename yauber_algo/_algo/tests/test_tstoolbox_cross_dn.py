import unittest
from yauber_algo.errors import *


class CrossDnTestCase(unittest.TestCase):
    def test_cross_dn(self):
        import yauber_algo.sanitychecks as sc

        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import cross_dn

        #
        # Function settings
        #
        algo = 'cross_dn'
        func = cross_dn

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan,  0, 0, 1, 0, 0, 0], dtype=np.float),
                func, (
                    np.array([1, 4, 5, 2, 4, 4, 4]),  # arr2
                    np.array([1, 2, 3, 3, 4, 4, 4]),  # arr_threshold

                ),
                suffix='cond'
            )

            s.check_regular(
                array([nan, 0, 0, 1, 0, 0, 0], dtype=np.float),
                func, (
                    np.array([1, 2, 5, 2, 4, 4, 4]),  # arr2

                    3,  # arr_threshold
                ),
                suffix='number_threshold'
            )

            s.check_regular(
                array([nan, 0, 0, 0, 0, 0, 0], dtype=np.float),
                func, (
                    np.array([1, 2, 3, 4, 4, 4, 4]),  # arr2

                    np.array([1, 2, 3, 3, 4, 4, 4]),  # arr_threshold
                ),
                suffix='equal_is_ignored'
            )

            s.check_regular(
                array([nan, 0, 0, 0, 0, 0, 1], dtype=np.float),
                func, (
                    np.array([1, 2, 2, 4, 4, 5, 2]),  # arr2

                    np.array([1, 2, 3, 3, 4, 4, 4]),  # arr_threshold
                ),
                suffix='crossup_is_ignored'
            )

            s.check_regular(
                array([nan, 0, 0, nan, nan, 0, 1], dtype=np.float),
                func, (
                    np.array([1, 2, 2, 4, 4, 5, 2]),  # arr2

                    np.array([1, 2, 3, nan, 4, 4, 4]),  # arr_threshold
                ),
                suffix='nan_threshold'
            )

            s.check_regular(
                array([nan, 0, 0, nan, nan, 0, 1], dtype=np.float),
                func, (
                    np.array([1, 2, 2, nan, 4, 5, 2]),  # arr2

                    np.array([1, 2, 3, 3, 4, 4, 4]),  # arr_threshold
                ),
                suffix='nan_array'
            )

            s.check_naninf(
                array([nan, 0, 0, nan, nan, 0, nan], dtype=np.float),
                func, (
                    np.array([1, 2, 2, inf, 4, 5, nan]),  # arr2

                    np.array([1, 2, 3, 3, 4, 4, 4]),  # arr_threshold
                ),
                suffix='inf_array'
            )

            s.check_naninf(
                array([nan, 0, 0, nan, nan, 0, nan], dtype=np.float),
                func, (
                    np.array([1, 2, 2, 3, 4, 5, 2]),  # arr2

                    np.array([1, 2, 3, inf, 4, 4, nan]),  # arr_threshold
                ),
                suffix='inf_threshold'
            )

            s.check_series(
                pd.Series(array([nan, 0, 0, 1, 0, 0, 0], dtype=np.float)),
                func, (
                    pd.Series(np.array([1, 2, 5, 2, 4, 4, 4])),  # arr2

                    pd.Series(np.array([1, 2, 3, 3, 4, 4, 4])),  # arr_threshold
                ),
                suffix='cond'
            )

            s.check_dtype_float(
                array([nan, 0, 0, 1, 0, 0, 0], dtype=np.float),
                func, (
                    np.array([1, 2, 5, 2, 4, 4, 4], dtype=np.float),  # arr2

                    np.array([1, 2, 3, 3, 4, 4, 4], dtype=np.float),  # arr_threshold
                ),
                suffix='cond'
            )

            s.check_dtype_int(
                array([nan, 0, 0, 1, 0, 0, 0], dtype=np.float),
                func, (
                    np.array([1, 2, 5, 2, 4, 4, 4], dtype=np.int32),  # arr2

                    np.array([1, 2, 3, 3, 4, 4, 4], dtype=np.int32),  # arr_threshold
                ),
                suffix='cond'
            )

            s.check_dtype_bool(
                array([nan, 0, 0, 1, 0, 0, 1], dtype=np.float),
                func, (
                    np.array([1, 1, 1, 0, 1, 1, 0], dtype=np.bool),  # arr2

                    np.array([1, 0, 0, 1, 1, 0, 1], dtype=np.bool),  # arr_threshold
                ),
                suffix='cond'
            )

            s.check_dtype_object(
                func, (
                    np.array([1, 1, 1, 0, 1, 1, 0], dtype=np.object),  # arr2

                    np.array([1, 0, 1, 0, 1, 0, 1], dtype=np.object),  # arr_threshold
                ),
                suffix='cond'
            )

            s.check_futref(5, 1,
               func, (
                               np.array([1, 2, 2, 4, 4, 4, 4, 5, 7, 8]),  # arr2

                               np.array([1, 2, 3, 3, 4, 4, 4, 2, 5, 6]),  # arr_threshold
               ),
               suffix='cond',
               min_checks_count=3,
            )

            s.check_window_consistency(5, 1,
                           func, (
                                           np.array([1, 2, 2, 4, 4, 4, 4, 5, 7, 8]),  # arr2

                                           np.array([1, 2, 3, 3, 4, 4, 4, 2, 5, 6]),  # arr_threshold
                           ),
                           suffix='cond',
                           min_checks_count=3,
                           )