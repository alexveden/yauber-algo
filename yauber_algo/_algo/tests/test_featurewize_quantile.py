import unittest
from yauber_algo.errors import *


class QuantileTestCase(unittest.TestCase):
    def test_quantile(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import quantile

        #
        # Function settings
        #
        algo = 'quantile'
        func = quantile

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, nan, nan, nan, 3, 4, 5]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    0.5
                ),
                suffix='reg'
            )

            s.check_regular(
                array([nan, nan, nan, nan, 3, 4, 7]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1])
                ),
                suffix='quantile_arr'
            )

            s.check_regular(
                array([nan, nan, nan, nan, 3, 4, nan]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, nan])
                ),
                suffix='quantile_arr_nan'
            )

            s.check_regular(
                array([nan, nan, nan, nan, 3, 4, nan]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, inf])
                ),
                suffix='quantile_arr_inf',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, nan, 3, 4, nan]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.00001])
                ),
                suffix='quantile_q_out_of_bounds_up',
                exception = YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, nan, 3, 4, nan]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.000001])
                ),
                suffix='quantile_q_out_of_bounds_dn',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, nan]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    nan
                ),
                suffix='reg_quantile_nan'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, 3.5, 4.5]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    6,
                    0.5
                ),
                suffix='midpoint'
            )

            s.check_regular(
                array([nan, nan, nan, nan, 3, 4, 5]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    4,
                    0.5
                ),
                suffix='min_period_error',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, 4, 5]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    -0.00001
                ),
                suffix='q_bounds_check_lower',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, 4, 5]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    1.00001
                ),
                suffix='q_bounds_check_upper',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, 4, 5]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    0,
                    0.5
                ),
                suffix='negative_per',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, nan, 5, 6, 7]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    1.0
                ),
                suffix='q_top'
            )

            s.check_regular(
                array([nan, nan, nan, nan, 1, 2, 3]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    0.0
                ),
                suffix='q_bot'
            )

            s.check_regular(
                array([nan, nan, nan, nan, 4.5, 5.5, 6.5]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7]),
                    5,
                    0.95
                ),
                suffix='q_95_midpoint'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, 4, 5, 6, 7]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    7,
                    0.5
                ),
                suffix='reg10'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, 5, 5.5, 6, 7]),
                func,
                (
                    array([nan, nan, 3, 4, 5, 6, 7, 8, 9, 10]),
                    7,
                    0.5
                ),
                suffix='skip_nan_in_quantile_calcs'
            )

            s.check_naninf(
                array([nan, nan, nan, nan, nan, nan, 4.5, 5, 6, nan]),
                func,
                (
                    array([inf, 2, 3, 4, 5, 6, 7, 8, 9, nan]),
                    7,
                    0.5
                ),
                suffix='nan_last'
            )

            s.check_dtype_float(
                array([nan, nan, nan, nan, 3, 4, 5], dtype=np.float),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7], dtype=np.float),
                    5,
                    0.5
                ),
                suffix=''
            )

            s.check_dtype_int(
                array([nan, nan, nan, nan, 3, 4, 5], dtype=np.float),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
                    5,
                    0.5
                ),
                suffix=''
            )

            s.check_dtype_bool(
                array([nan, nan, nan, nan, 1, 0, 1], dtype=np.float),
                func,
                (
                    array([1, 0, 1, 0, 1, 0, 1], dtype=np.bool),
                    5,
                    0.5
                ),
                suffix='',
                exception=YaUberAlgoDtypeNotSupportedError

            )

            s.check_dtype_bool(
                array([nan, nan, nan, nan, nan, .5, .5], dtype=np.float),
                func,
                (
                    array([1, 0, 1, 0, 1, 0, 1], dtype=np.bool),
                    6,
                    0.5
                ),
                suffix='midpoint',
                exception=YaUberAlgoDtypeNotSupportedError
            )

            s.check_dtype_object(
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 7], dtype=np.object),
                    5,
                    0.5
                ),
                suffix=''
            )

            s.check_series(
                pd.Series(array([nan, nan, nan, nan, 3, 4, 5])),
                func,
                (
                    pd.Series(array([1, 2, 3, 4, 5, 6, 7])),
                    5,
                    0.5
                ),
                suffix='reg'
            )

            s.check_futref(5, 1,
                           func,
                           (
                               np.random.random(1000),
                               50,
                               0.5

                           ),
                           )

            s.check_window_consistency(5, 1,
                                       func,
                                       (
                                           np.random.random(1000),
                                           50,
                                           0.5
                                       ),
                                       )

