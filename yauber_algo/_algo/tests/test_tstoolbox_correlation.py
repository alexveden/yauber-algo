import unittest
from yauber_algo.errors import *


class CorrelationTestCase(unittest.TestCase):
    def test_correlation(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import correlation

        #
        # Function settings
        #
        algo = 'correlation'
        func = correlation

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, nan, nan, nan, 0.43001307, 0.30197859, -0.07360677]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 4, 2, 7, 1, 8, 9]),
                    5
                ),
                suffix='reg'
            )

            s.check_regular(
                array([nan, nan, nan, nan, 0.43001307, 0.30197859, -0.07360677]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 4, 2, 7, 1, 8, 9]),
                    0
                ),
                suffix='period_zero',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, nan, 0.43001307, 0.30197859, -0.07360677]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 4, 2, 7, 1, 8, 9]),
                    -1
                ),
                suffix='period_negative',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, nan, 0.43001307, 0.30197859, -0.07360677]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 4, 2, 7, 1, 8, 9]),
                    4
                ),
                suffix='period_less_5',
                exception=YaUberAlgoInternalError
            )

            # The correlation is exactly the same to Pandas x.rolling(N).corr(y)
            x = pd.Series(np.random.random(1000))
            y = pd.Series(np.random.random(1000))
            s.check_series(
                x.rolling(20).corr(y),
                func,
                (
                    x,
                    y,
                    20
                ),
                suffix='pandas'
            )

            s.check_naninf(
                array([nan, nan, nan, nan, 0.43001307, nan, nan]),
                func,
                (
                    array([3, 2, 1, 4, 3, nan, inf]),
                    array([1, 4, 2, 7, 1, 8, 9]),
                    5
                ),
                suffix='x'
            )

            s.check_naninf(
                array([nan, nan, nan, nan, 0.43001307, nan, nan]),
                func,
                (
                    array([3, 2, 1, 4, 3, 8, 9]),
                    array([1, 4, 2, 7, 1, nan, inf]),
                    5
                ),
                suffix='y'
            )

            s.check_dtype_float(
                array([nan, nan, nan, nan, 0.43001307, 0.30197859, -0.07360677], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.float),
                    array([1, 4, 2, 7, 1, 8, 9], dtype=np.float),
                    5
                ),
                suffix='reg'
            )

            s.check_dtype_int(
                array([nan, nan, nan, nan, 0.43001307, 0.30197859, -0.07360677], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.int32),
                    array([1, 4, 2, 7, 1, 8, 9], dtype=np.int32),
                    5
                ),
                suffix='reg'
            )

            s.check_dtype_bool(
                array([nan, nan, nan, nan, -0.66666667, -0.61237244, -0.61237244], dtype=np.float),
                func,
                (
                    array([0, 1, 1, 1, 0, 1, 1], dtype=np.bool),
                    array([1, 1, 0, 0, 1, 0, 1], dtype=np.bool),
                    5
                ),
                suffix='reg'
            )

            s.check_dtype_object(
                func,
                (
                    array([0, 1, 1, 1, 0, 1, 1], dtype=np.object),
                    array([1, 1, 0, 0, 1, 0, 1], dtype=np.bool),
                    5
                ),
                suffix='reg'
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(100),
                               np.random.random(100),
                               5
                           ),
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(100),
                                           np.random.random(100),
                                           5
                                       ),
                                       )

