import unittest
from yauber_algo.errors import *


class TWMATestCase(unittest.TestCase):
    def test_twma(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import twma

        #
        # Function settings
        #
        algo = 'twma'
        func = twma

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, nan, 2, 7/3]),
                func,
                (
                    array([3, 2, 1, 4]),
                    np.array([1, 1, 1])
                ),
                suffix='twma_equal_weight'
            )

            s.check_regular(
                array([nan, nan, (1*1+2*0.5+3*0.25)/1.75, (4*1+1*0.5+2*0.25)/1.75]),
                func,
                (
                    array([3, 2, 1, 4]),
                    np.array([1, 0.5, 0.25])
                ),
                suffix='twma_linear_weight'
            )

            s.check_regular(
                array([nan, nan, 2, 7 / 3]),
                func,
                (
                    array([3, 2, 1, 4]),
                    [1, 1, 1]
                ),
                suffix='twma_list_weight'
            )
            s.check_regular(
                array([nan, nan, 2, 7 / 3]),
                func,
                (
                    array([3, 2, 1, 4]),
                    pd.Series([1, 1, 1])
                ),
                suffix='twma_series_weight'
            )

            s.check_regular(
                array([nan, nan, 2, 7 / 3]),
                func,
                (
                    array([3, 2, 1, 4]),
                    [1, 1, 1, 2, 2]
                ),
                suffix='twma_weight_gt_ser',
                exception=YaUberAlgoArgumentError,
            )

            s.check_regular(
                array([nan, nan, nan, nan]),
                func,
                (
                    array([3, 2, 1, 4]),
                    np.array([0, 0, 0])
                ),
                suffix='twma_zeroweight'
            )

            s.check_regular(
                array([nan, nan, 2, 7 / 3]),
                func,
                (
                    array([3, 2, 1, 4]),
                    np.array([1, 1, nan])
                ),
                suffix='twma_nan_weight',
                exception=YaUberAlgoArgumentError,
            )

            s.check_naninf(
                array([nan, nan, nan, nan]),
                func,
                (
                    array([3, 2, nan, inf]),
                    np.array([1, 1, 1, 1])
                ),
                suffix='',
            )

            s.check_series(
                pd.Series(array([nan, nan, 2, 7 / 3])),
                func,
                (
                    pd.Series(array([3, 2, 1, 4])),
                    np.array([1, 1, 1])
                ),
            )

            s.check_dtype_float(
                array([nan, nan, 2, 7 / 3], dtype=float),
                func,
                (
                    array([3, 2, 1, 4], dtype=float),
                    np.array([1, 1, 1], dtype=float)
                ),
            )

            s.check_dtype_bool(
                array([nan, nan, 1/3, 2 / 3], dtype=float),
                func,
                (
                    array([0, 1, 0, 1], dtype=bool),
                    np.array([1, 1, 1], dtype=float)
                ),
            )
            s.check_dtype_int(
                array([nan, nan, 2, 7 / 3], dtype=float),
                func,
                (
                    array([3, 2, 1, 4], dtype=np.int32),
                    np.array([1, 1, 1], dtype=np.int32)
                ),
            )

            s.check_dtype_object(
                func,
                (
                    array([3, 2, 1, 4], dtype=np.object),
                    np.array([1, 1, 1], dtype=float)
                ),
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(100),
                               np.array([1, 0.5, 0.25, 0.2, 0.1]),
                           ),
                           fix_args=[1], # Use weights args as is
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(100),
                                           np.array([1, 0.5, 0.25, 0.2, 0.1]),
                                       ),
                                       fix_args=[1],  # Use weights args as is
                                       )