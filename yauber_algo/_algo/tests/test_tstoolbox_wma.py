import unittest
from yauber_algo.errors import *


class WMATestCase(unittest.TestCase):
    def test_wma(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import wma

        #
        # Function settings
        #
        algo = 'wma'
        func = wma

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]) / 3.0,
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 1, 1, 1, 1, 1, 1]),
                    3
                ),
                suffix='wma_equal_weight'
            )

            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]) / 3.0,
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 1, 1, 1, 1, 1, 1]),
                    0
                ),
                suffix='period_zero',
                exception=YaUberAlgoArgumentError,
            )

            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]) / 3.0,
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 1, 1, 1, 1, 1, 1]),
                    -1
                ),
                suffix='period_negative',
                exception=YaUberAlgoArgumentError,
            )

            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]) / 3.0,
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([-1, 1, 1, 1, 1, 1, 1]),
                    3
                ),
                suffix='negative_weight',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([nan, nan, 6, 7, 8, 9, 6]) / 3.0,
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 1, 1, 1, 1, 1, -1]),
                    3
                ),
                suffix='negative_weight2',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, nan]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([0, 0, 0, 0, 0, 0, 0]),
                    3
                ),
                suffix='zero_weight'
            )

            s.check_regular(
                array([nan, nan, (3*1+2*2+1*3)/6, (2*2+1*3+4*1)/6, (1*3+4+3)/5, (4*1+3*1+2*1)/3, (3*1+2*1+1*1)/3]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 2, 3, 1, 1, 1, 1]),
                    3
                ),
                suffix='wma_weight_applied'
            )

            s.check_naninf(
                array([nan, nan, 6, 7, 8, nan, nan]) / 3.0,
                func,
                (
                    array([3, 2, 1, 4, 3, inf, nan]),
                    array([1, 1, 1, 1, 1, 1, 1]),
                    3
                ),
                suffix='wma_arr_nan'
            )

            s.check_naninf(
                array([nan, nan, 6, 7, 8, nan, nan]) / 3.0,
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    array([1, 1, 1, 1, 1, nan, inf]),
                    3
                ),
                suffix='wma_weight_nan'
            )

            s.check_series(
                pd.Series(array([nan, nan, 6, 7, 8, 9, 6]) / 3.0),
                func,
                (
                    pd.Series(array([3, 2, 1, 4, 3, 2, 1])),
                    pd.Series(array([1, 1, 1, 1, 1, 1, 1])),
                    3
                ),
                suffix=''
            )

            s.check_dtype_float(
                array([nan, nan, 6, 7, 8, 9, 6], dtype=float) / 3.0,
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=float),
                    array([1, 1, 1, 1, 1, 1, 1], dtype=float),
                    3
                ),
                suffix=''
            )

            s.check_dtype_int(
                array([nan, nan, 6, 7, 8, 9, 6], dtype=float) / 3.0,
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.int32),
                    array([1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
                    3
                ),
                suffix=''
            )

            s.check_dtype_bool(
                array([nan, nan, 6, 7, 8, 9, 6], dtype=float) / 3.0,
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=float),
                    array([1, 1, 1, 1, 1, 1, 1], dtype=bool),
                    3
                ),
                suffix=''
            )

            s.check_dtype_object(
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.object),
                    array([1, 1, 1, 1, 1, 1, 1], dtype=bool),
                    3
                ),
                suffix=''
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
