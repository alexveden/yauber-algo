import unittest
from yauber_algo.errors import *


class HHVTestCase(unittest.TestCase):
    def test_hhv(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import hhv

        #
        # Function settings
        #
        algo = 'hhv'
        func = hhv

        hhv_arr3 = array([0, 0, nan, 1])

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, nan, 7, 6, 5, 4, 3]),
                func,
                (
                    array([7, 6, 5, 4, 3, 2, 1]),
                    3
                ),
                suffix='decreasing'
            )

            s.check_regular(
                array([nan, nan, 7, 6, 5, 4, 3]),
                func,
                (
                    array([7, 6, 5, 4, 3, 2, 1]),
                    -3
                ),
                suffix='negative_period',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, 3, 4, 5, 6, 7]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    3
                ),
                suffix='increasing'
            )

            s.check_regular(
                array([nan, nan, nan, 1]),
                func,
                (
                    array([0, 0, nan, 1]),
                    3
                ),
                suffix='with_nan'
            )

            s.check_regular(
                array([nan, nan, nan, 1]),  # Expected
                func,
                (
                    array([nan, nan, nan, 1]),  # Input array
                    3
                ),
                suffix='with_nan2'
            )

            s.check_regular(
                array([nan, nan, nan, nan]),  # Expected
                func,
                (
                    array([0, nan, nan, nan]),  # Input array
                    3
                ),
                suffix='with_nan3'
            )

            s.check_naninf(
                array([nan, nan, nan, nan]),  # Expected
                func,
                (
                    array([inf, 0, -inf, nan]),  # Input array
                    3
                ),
                suffix='pos_inf'
            )

            s.check_naninf(
                array([nan, nan, nan, nan]),  # Expected
                func,
                (
                    array([-inf, 0, inf, nan]),  # Input array
                    3
                ),
                suffix='neg_inf'
            )

            s.check_series(
                pd.Series(array([nan, nan, 3, 4, 5, 6, 7])),
                func,
                (
                    pd.Series(array([1, 2, 3, 4, 5, 6, 7])),
                    3
                ),
            )

            s.check_dtype_float(
                array([nan, nan, 3, 4, 5, 6, 7], dtype=np.float),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7], dtype=np.float),
                    3
                ),
            )

            s.check_dtype_bool(
                array([nan, nan, 1, 1, 1, 0, 1], dtype=np.float),
                func,
                (
                    array([False, False, True, False, False, False, True], dtype=np.bool),
                    3
                ),
            )

            s.check_dtype_int(
                array([nan, nan, 3, 4, 5, 6, 7], dtype=np.float),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
                    3
                ),
            )

            s.check_dtype_object(
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7], dtype=np.object),
                    3
                ),
            )

            s.check_futref(5, 1,
                           func, (sc.SAMPLE_10_FLOAT, 3),
                           min_checks_count=3,
                           )

            s.check_window_consistency(5, 1,
                           func, (sc.SAMPLE_10_FLOAT, 3),
                           min_checks_count=3,
                           )
