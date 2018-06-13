import unittest
from yauber_algo.errors import *


class PercentRankTestCase(unittest.TestCase):
    def test_category(self):
        import yauber_algo.sanitychecks as sc
        from numpy import array, nan, inf
        import os
        import sys
        import pandas as pd
        import numpy as np

        from yauber_algo.algo import percent_rank

        #
        # Function settings
        #
        algo = 'percent_rank'
        func = percent_rank

        with sc.SanityChecker(algo) as s:
            #
            # Check regular algorithm logic
            #
            s.check_regular(
                array([nan, nan, nan, nan, nan, .30, .10]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    5
                ),
                suffix='reg'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, 1.00, .90]),
                func,
                (
                    array([1, 2, 3, 4, 5, 6, 6]),
                    5
                ),
                suffix='equal_numbers'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, .50, .50]),
                func,
                (
                    array([1, 1, 1, 1, 1, 1, 1]),
                    5
                ),
                suffix='all_equal_numbers'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, .10]),
                func,
                (
                    array([nan, 2, 1, 4, 3, 2, 1]),
                    5
                ),
                suffix='skip_nan'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, nan, nan]),
                func,
                (
                    array([nan, 2, nan, 2, 3, 2, 1]),
                    5
                ),
                suffix='skip_nan_min_count_5'
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, 2 / 5, 1 / 5]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    4
                ),
                suffix='min_period_eq_5',
                exception=YaUberAlgoInternalError
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, 2 / 5, 1 / 5]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    0
                ),
                suffix='zero_period_err',
                exception=YaUberAlgoArgumentError
            )

            s.check_regular(
                array([nan, nan, nan, nan, nan, 2 / 5, 1 / 5]),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1]),
                    -1
                ),
                suffix='neg_period_err',
                exception=YaUberAlgoArgumentError
            )

            s.check_naninf(
                array([nan, nan, nan, nan, nan, nan, .10, nan, .20]),
                func,
                (
                    array([nan, 2, 1, 4, 3, 5, 1, inf, 1]),
                    6
                ),
                suffix='inf'
            )

            s.check_naninf(
                array([nan, nan, nan, nan, nan, nan, .10, nan, nan]),
                func,
                (
                    array([nan, 2, 1, 4, 3, 5, 1, inf, nan]),
                    6
                ),
                suffix='inf_nan'
            )

            s.check_naninf(
                array([nan, nan, nan, nan, nan, nan, .10, nan, .20]),
                func,
                (
                    array([nan, 2, 1, 4, 3, 5, 1, -inf, 1]),
                    6
                ),
                suffix='neg_inf'
            )

            s.check_series(
                pd.Series(array([nan, nan, nan, nan, nan, .30, .10])),
                func,
                (
                    pd.Series(array([3, 2, 1, 4, 3, 2, 1])),
                    5
                ),
                suffix=''
            )

            s.check_dtype_float(
                array([nan, nan, nan, nan, nan, .30, .10], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.float),
                    5
                ),
                suffix=''
            )

            s.check_dtype_bool(
                array([nan, nan, nan, nan, nan, .20, .70], dtype=np.float),
                func,
                (
                    array([0, 1, 1, 0, 1, 0, 1], dtype=np.bool),
                    5
                ),
                suffix=''
            )

            s.check_dtype_int(
                array([nan, nan, nan, nan, nan, .30, .10], dtype=np.float),
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.int32),
                    5
                ),
                suffix=''
            )

            s.check_dtype_object(
                func,
                (
                    array([3, 2, 1, 4, 3, 2, 1], dtype=np.object),
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