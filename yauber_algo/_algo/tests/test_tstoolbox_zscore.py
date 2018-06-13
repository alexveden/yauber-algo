import unittest
from yauber_algo.errors import *


class zScoreTestCase(unittest.TestCase):
    def test_zscore(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import zscore

        #
        # Function settings
        #
        algo = 'zscore'
        func = zscore
        def zsc(arr):
            return (arr[-1] - np.mean(arr))/ np.std(arr)

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            a = array([3, 2, 1, 4, 3, 2, 1])
            s.check_regular(
                array([nan, nan, nan, nan, zsc(a[0:5]), zsc(a[1:6]), zsc(a[2:7])]),
                func,
                (
                    a,
                    5
                ),
                suffix='match_with_numpy'
            )

            a = array([1, 1, 1, 1, 1, 1, 1])
            s.check_regular(
                array([nan, nan, nan, nan, 0, 0, 0]),
                func,
                (
                    a,
                    5
                ),
                suffix='zscore_no_disp_is_zero'
            )


            a = array([3, 2, 1, 4, 3, 2, 1])
            s.check_regular(
                array([nan, nan, nan, nan, zsc(a[0:5]), zsc(a[1:6]), zsc(a[2:7])]),
                func,
                (
                    a,
                    -15
                ),
                suffix='negative_period',
                exception=YaUberAlgoArgumentError,
            )

            a = array([3, 2, 1, 4, 3, 2, 1])
            s.check_regular(
                array([nan, nan, nan, nan, zsc(a[0:5]), zsc(a[1:6]), zsc(a[2:7])]),
                func,
                (
                    a,
                    4
                ),
                suffix='period_less_5',
                exception=YaUberAlgoInternalError
            )

            a = array([nan, 2, 1, 4, 3, 2, nan, 2])
            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, nan, zsc(pd.Series(a).dropna().values)]),
                func,
                (
                    a,
                    7
                ),
                suffix='numpy_with_nan'
            )

            a = array([inf, 2, 1, 4, 3, 2, nan])
            s.check_naninf(
                array([nan, nan, nan, nan, nan, nan, nan]),
                func,
                (
                    a,
                    7
                ),
                suffix='numpy_with_nan'
            )

            a = array([3, 2, 1, 4, 3, 2, 1])
            s.check_series(
                pd.Series(array([nan, nan, nan, nan, zsc(a[0:5]), zsc(a[1:6]), zsc(a[2:7])])),
                func,
                (
                    pd.Series(a),
                    5
                ),
                suffix=''
            )

            a = array([3, 2, 1, 4, 3, 2, 1], dtype=np.float)
            s.check_dtype_float(
                array([nan, nan, nan, nan, zsc(a[0:5]), zsc(a[1:6]), zsc(a[2:7])], dtype=np.float),
                func,
                (
                    a,
                    5
                ),
                suffix=''
            )

            a = array([3, 2, 1, 4, 3, 2, 1], dtype=np.int32)
            s.check_dtype_int(
                array([nan, nan, nan, nan, zsc(a[0:5]), zsc(a[1:6]), zsc(a[2:7])], dtype=np.float),
                func,
                (
                    a,
                    5
                ),
                suffix=''
            )

            a = array([0, 1, 1, 0, 0, 1, 1], dtype=np.bool)
            s.check_dtype_bool(
                array([nan, nan, nan, nan, zsc(a[0:5]), zsc(a[1:6]), zsc(a[2:7])], dtype=np.float),
                func,
                (
                    a,
                    5
                ),
                suffix=''
            )

            a = array([0, 1, 1, 0, 0, 1, 1], dtype=np.object)
            s.check_dtype_object(
                func,
                (
                    a,
                    5
                ),
                suffix=''
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