import unittest
from yauber_algo.errors import *


class SumTestCase(unittest.TestCase):
    def test_sum(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import sum

        #
        # Function settings
        #
        algo = 'sum'
        func = sum

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    3
                ),
                suffix='regular_sum'
            )

            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    -1,
                ),
                suffix='regular_neg_period',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    0,
                ),
                suffix='regular_zero_period',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, 6, 7, 9, 6]),
                func,
                (
                    array([3, 2, nan, 4, 3, 2, 1]),
                    3
                ),
                suffix='regular_nan_skipped'
            )

            s.check_regular(
                array([nan, nan, 2, nan, nan, nan, nan]),
                func,
                (
                    array([nan, nan, 2, nan, nan, nan, nan]),
                    3
                ),
                suffix='regular_nan_filled'
            )

            s.check_regular(
                array([nan, nan, nan, 4, 7, 9, 6]),
                func,
                (
                    array([nan, nan, nan, 4, 3, 2, 1]),
                    3
                ),
                suffix='regular_nan_skipped2'
            )

            s.check_naninf(
                array([nan, nan, nan, 4, 7, 9, 6]),
                func,
                (
                    array([nan, inf, -inf, 4, 3, 2, 1]),
                    3
                ),
                suffix=''
            )

            s.check_naninf(
                array([nan, nan, nan, 4, 7, 9, nan]),
                func,
                (
                    array([nan, inf, -inf, 4, 3, 2, nan]),
                    3
                ),
                suffix='codex_20180209'
            )

            s.check_series(
                pd.Series(array([nan, nan, 6, 7, 8, 9, 6])),
                func,
                (
                    pd.Series(array([3, 2, 1, 4, 3, 2, 1])),
                    3
                ),
                suffix='ser'
            )

            s.check_dtype_float(
                array([nan, nan, 6, 7, 8, 9, 6], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.float),
                    3
                ),
            )

            s.check_dtype_bool(
                array([nan, nan, 2, 2, 1, 1, 1], dtype=np.float),
                func,
                (
                    array([0, 1, 1, 0, 0, 1, 0], dtype=np.bool),
                    3
                ),
            )

            s.check_dtype_int(
                array([nan, nan, 6, 7, 8, 9, 6], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.int32),
                    3
                ),
            )

            s.check_dtype_object(
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.object),
                    3
                ),
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(100),
                               5
                           ),
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(100),
                                           5
                                       ),
                                       )
